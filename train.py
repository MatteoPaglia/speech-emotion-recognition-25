import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
import wandb
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import argparse

import config as Config
from dataset.custom_ravdess_dataset import CustomRAVDESSDataset
from models import get_model
from utils.training_utils import SimpleEarlyStopping, save_swa_checkpoint, update_bn_custom


# --- 1. ARGPARSE (SCELTA MODELLO) ---
parser = argparse.ArgumentParser(description='Train Speech Emotion Recognition Model')
parser.add_argument('--model', type=str, default='CRNN_BiLSTM', 
                    choices=['CRNN_BiLSTM', 'CRNN_BiGRU'],
                    help='Tipo di modello da utilizzare (default: CRNN_BiLSTM)')
args = parser.parse_args()

MODEL_TYPE = args.model
# --- 2. CONFIGURAZIONE (IMPORTATE DA CONFIG) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Crea cartella checkpoints se non esiste
Path("checkpoints").mkdir(exist_ok=True)

# Importa da config
BATCH_SIZE = Config.BATCH_SIZE_RAVDESS
LEARNING_RATE = Config.LEARNING_RATE_RAVDESS
NUM_EPOCHS = Config.NUM_EPOCHS_RAVDESS
NUM_CLASSES = Config.NUM_CLASSES_RAVDESS
TIME_STEPS = Config.TIME_STEPS_RAVDESS
MEL_BANDS = Config.MEL_BANDS_RAVDESS
DROPOUT = Config.DROPOUT_RAVDESS
SPEC_FREQ_MASK = Config.SPEC_FREQ_MASK_RAVDESS
SPEC_TIME_MASK = Config.SPEC_TIME_MASK_RAVDESS
CLASS_WEIGHTS = Config.CLASS_WEIGHTS_RAVDESS
SWA_START_EPOCH = Config.SWA_START_EPOCH_RAVDESS
SWA_LR = Config.SWA_LR_RAVDESS
WEIGHT_DECAY = Config.WEIGHT_DECAY_RAVDESS


# --- 4. FUNZIONE DI TRAINING ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()  
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in loader:
        data = batch['audio_features'].to(device)
        targets = batch['emotion_id'].to(device)

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predictions = scores.max(1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy

# --- 5. FUNZIONE DI VALIDATION ---
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            data = batch['audio_features'].to(device)
            targets = batch['emotion_id'].to(device)

            scores = model(data)
            loss = criterion(scores, targets)

            running_loss += loss.item()
            _, predictions = scores.max(1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy

# --- 6. MAIN LOOP ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    #ravdess_path = Config.COLAB_RAVDESS_PATH #for colab training
    ravdess_path = Path(Config.RAVDESS_PATH) #for local training
    
    if not ravdess_path or not ravdess_path.exists():
        raise ValueError("❌ RAVDESS non trovato! Verifica il percorso del dataset.")
    
    print(f"\n✅ RAVDESS trovato: {ravdess_path}\n")
    
    # Create RAVDESS datasets
    train_RAVDESS_dataset = CustomRAVDESSDataset(
        dataset_root=str(ravdess_path), 
        split='train',
        spec_freq_mask=SPEC_FREQ_MASK,
        spec_time_mask=SPEC_TIME_MASK
    )
    val_RAVDESS_dataset = CustomRAVDESSDataset(dataset_root=str(ravdess_path), split='validation')
    
    print(f"Train samples: {len(train_RAVDESS_dataset)}")
    print(f"Val samples: {len(val_RAVDESS_dataset)}")

    # Create RAVDESS DataLoaders
    
    train_RAVDESS_dataloader = DataLoader(train_RAVDESS_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_RAVDESS_dataloader = DataLoader(val_RAVDESS_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Inizializzazione Modello
    model = get_model(MODEL_TYPE, batch_size=BATCH_SIZE, time_steps=TIME_STEPS, dropout=DROPOUT).to(DEVICE)
    
    # Stampa dell'architettura del modello
    print("\n" + "="*80)
    print("🏗️ ARCHITETTURA DEL MODELLO")
    print("="*80)
    print(model)
    print("="*80 + "\n")

    # Class weights per bilanciare le classi
    # Ordine delle classi (da EMOTION_ID_MAP in custom_ravdess_dataset.py):
    # 0: Neutral | 1: Happy | 2: Sad | 3: Angry
    # Sad ha peso maggiore perché è la classe più difficile da riconoscere
    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)
    # Normalizza i pesi (somma = 1)
    class_weights = class_weights / class_weights.sum()

    # CrossEntropyLoss con class weights per bilanciare il dataset
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Adam optimizer con weight decay aumentato per ridurre overfitting e oscillazioni
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # AGGIUNTA: Scheduler per ridurre il LR quando la loss si appiattisce
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,     # Dimezza il LR
        patience=3      # Se non migliora per 3 epoche
    )

    # AGGIUNTA: Stochastic Weight Averaging (SWA)
    # SWA calcola la media dei pesi del modello durante il training
    # Questo porta a modelli con migliore generalizzazione
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)

    # Ciclo delle Epoche
    best_val_acc = 0.0
    best_swa_val_acc = 0.0
    early_stopping = SimpleEarlyStopping(patience=10)
    using_swa = False  # Flag per indicare se siamo nella fase SWA

    # Genera timestamp per il run (ora italiana UTC+1)
    timestamp = (datetime.now() + timedelta(hours=1)).strftime("%Y%m%d_%H%M%S")

    # --- INIZIALIZZA WANDB ---
    wandb.init(
        project="speech-emotion-recognition",
        name=f"train_{timestamp}",
        config={
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "num_classes": NUM_CLASSES,
            "time_steps": TIME_STEPS,
            "mel_bands": MEL_BANDS,
            "architecture": MODEL_TYPE,
            "dataset": "RAVDESS",
            "optimizer": "Adam",
            "weight_decay": WEIGHT_DECAY,
            "loss_function": "CrossEntropyLoss",
            "early_stopping_patience": early_stopping.patience,
            "device": str(DEVICE),
            "swa_start_epoch": SWA_START_EPOCH,
            "swa_lr": SWA_LR,
            # Model hyperparameters
            "dropout": DROPOUT,
            # Augmentation hyperparameters
            "spec_freq_mask": SPEC_FREQ_MASK,
            "spec_time_mask": SPEC_TIME_MASK,
            # Class weights
            "class_weights": CLASS_WEIGHTS
        }
    )

    # --- STAMPA IPERPARAMETRI ---
    print("\n" + "="*80)
    print("🔧 IPERPARAMETRI DI TRAINING")
    print("="*80)
    print(f"Device:                {DEVICE}")
    print(f"Batch Size:            {BATCH_SIZE}")
    print(f"Learning Rate:         {LEARNING_RATE}")
    print(f"Weight Decay (L2):     {WEIGHT_DECAY}")
    print(f"Number of Epochs:      {NUM_EPOCHS}")
    print(f"Early Stopping Patience: {early_stopping.patience}")
    print(f"\nModello:")
    print(f"  - Num Classes:       {NUM_CLASSES}")
    print(f"  - Time Steps:        {TIME_STEPS}")
    print(f"  - Mel Bands:         {MEL_BANDS}")
    print(f"\nOptimizer:             Adam")
    print(f"Loss Function:         CrossEntropyLoss")
    print(f"Train Samples:         {len(train_RAVDESS_dataset)}")
    print(f"Val Samples:           {len(val_RAVDESS_dataset)}")
    print("="*80 + "\n")

    print("\n" + "="*80)
    print(f"{'Epoch':<8} {'Train Loss':<15} {'Train Acc':<15} {'Val Loss':<15} {'Val Acc':<15}")
    print("="*80)
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_RAVDESS_dataloader, criterion, optimizer, DEVICE)
        
        # Validation
        val_loss, val_acc = validate(model, val_RAVDESS_dataloader, criterion, DEVICE)

        # Stampa compatta di questa epoca
        epoch_marker = ""
        if val_acc > best_val_acc:
            epoch_marker = "⭐"
            best_val_acc = val_acc
            checkpoint_path = Path(__file__).parent / "checkpoints" / "best_model.pth"
            torch.save(model.state_dict(), str(checkpoint_path))
        
        print(f"{epoch+1:<8} {train_loss:<15.4f} {train_acc:<15.2f}% {val_loss:<15.4f} {val_acc:<15.2f}% {epoch_marker}")

        # LOGICA SWA
        if epoch >= SWA_START_EPOCH:
            if not using_swa:
                print(f"\n🔄 SWA attivato dalla epoca {epoch+1}\n")
                using_swa = True
            
            swa_model.update_parameters(model)
            swa_scheduler.step()
            
            if (epoch + 1) % 5 == 0:
                update_bn_custom(train_RAVDESS_dataloader, swa_model, DEVICE)
                swa_val_loss, swa_val_acc = validate(swa_model, val_RAVDESS_dataloader, criterion, DEVICE)
                print(f"  SWA: {swa_val_loss:.4f} loss | {swa_val_acc:.2f}% acc")
                
                wandb.log({
                    "swa_val_loss": swa_val_loss,
                    "swa_val_accuracy": swa_val_acc
                })
                
                if swa_val_acc > best_swa_val_acc:
                    best_swa_val_acc = swa_val_acc
                    swa_checkpoint_path = Path(__file__).parent / "checkpoints" / "best_swa_model.pth"
                    save_swa_checkpoint(swa_model, str(swa_checkpoint_path))
        else:
            scheduler.step(val_loss)

        # Log metriche su W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        # Early Stopping
        early_stopping.step(val_loss)
        if early_stopping.should_stop:
            print(f"\n⏹️ Early stopping alla epoca {epoch+1}")
            break

    print("="*80)
    
    # Valutazione finale del SWA model
    if using_swa:
        print(f"\n🔄 Valutazione SWA finale...")
        update_bn_custom(train_RAVDESS_dataloader, swa_model, DEVICE)
        final_swa_val_loss, final_swa_val_acc = validate(swa_model, val_RAVDESS_dataloader, criterion, DEVICE)
        print(f"  Final SWA: {final_swa_val_loss:.4f} loss | {final_swa_val_acc:.2f}% acc")
        
        if final_swa_val_acc > best_swa_val_acc:
            swa_checkpoint_path = Path(__file__).parent / "checkpoints" / "best_swa_model.pth"
            save_swa_checkpoint(swa_model, str(swa_checkpoint_path))
            best_swa_val_acc = final_swa_val_acc

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED")
    print("="*80)
    print(f"Best Validation Accuracy (Regular):  {best_val_acc:.2f}%")
    if using_swa:
        print(f"Best Validation Accuracy (SWA):      {best_swa_val_acc:.2f}%")
        print(f"SWA Improvement:                     {(best_swa_val_acc - best_val_acc):+.2f}%")
        print(f"\n📦 Recommended checkpoint: best_swa_model.pth")
    else:
        print(f"\n📦 Recommended checkpoint: best_model.pth")
    print("="*80 + "\n")

    wandb.finish()