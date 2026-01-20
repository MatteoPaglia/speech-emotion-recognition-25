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

from dataset.custom_ravdess_dataset import CustomRAVDESSDataset
from models import get_model
import argparse

# --- 1. ARGPARSE (SCELTA MODELLO) ---
parser = argparse.ArgumentParser(description='Train Speech Emotion Recognition Model')
parser.add_argument('--model', type=str, default='CRNN_BiLSTM', 
                    choices=['CRNN_BiLSTM', 'CRNN_BiGRU'],
                    help='Tipo di modello da utilizzare (default: CRNN_BiLSTM)')
args = parser.parse_args()

MODEL_TYPE = args.model
# --- 2. CONFIGURAZIONE (IPERPARAMETRI) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Se usi Mac M1/M2 puoi usare: torch.device("mps")

# Crea cartella checkpoints se non esiste
Path("checkpoints").mkdir(exist_ok=True)

BATCH_SIZE = 32
LEARNING_RATE = 0.0005  # Adam lavora bene con 1e-3 o 1e-4
NUM_EPOCHS = 100
NUM_CLASSES = 4       # We consider only 4 emotions: Neutral, Happy, Sad, Angry
TIME_STEPS = 200      # Consider avg or max time steps calculated before 
MEL_BANDS = 128

# Model Configuration
DROPOUT = 0.4  # Ridotto da 0.6 per permettere di imparare feature sottili (es. Sad)

# Augmentation Configuration
SPEC_FREQ_MASK = 12  # Ridotto da 18 per preservare feature sottili
SPEC_TIME_MASK = 15  # Ridotto da 25 per preservare feature sottili

# Class Weights Configuration (Neutral, Happy, Sad, Angry)
CLASS_WEIGHTS = [1.0, 1.0, 1.5, 1.0]  # Sad ha peso maggiore perché più difficile

# SWA Configuration
SWA_START_EPOCH = 15  # Inizia SWA dopo 15 epoche (quando il modello è già convergente)
SWA_LR = 0.0001       # Learning rate costante per SWA (più basso del LR iniziale)

# --- 2. CONFIGURAZIONE CLASSE PER EARLY STOPPING ---
class SimpleEarlyStopping:
    """Versione semplice: si ferma appena la validation loss smette di migliorare."""
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            print(f"📊 Best val loss: {val_loss:.4f}")
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            print(f"✓ Migliorato! New best: {val_loss:.4f}")
        else:
            self.counter += 1
            print(f"⚠️ Nessun miglioramento ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"🛑 STOP! Nessun miglioramento per {self.patience} epoche")

print("✓ Classe SimpleEarlyStopping pronta!")

# --- 3. HELPER FUNCTIONS ---
def save_swa_checkpoint(swa_model, path):
    """
    Salva il checkpoint SWA estraendo i pesi dal wrapper AveragedModel.
    Questo garantisce compatibilità per il caricamento futuro (es. Noisy Student).
    """
    unwrapped_state_dict = swa_model.module.state_dict()
    torch.save(unwrapped_state_dict, path)

def update_bn_custom(loader, model, device):
    """
    Wrapper per update_bn che gestisce il formato custom del nostro dataloader.
    Il nostro loader restituisce dizionari {'audio_features': tensor, ...}
    mentre update_bn si aspetta direttamente i tensori.
    """
    model.train()
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum
    
    if not momenta:
        return
    
    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0
    
    for batch in loader:
        # Estrai il tensore audio dal dizionario
        inputs = batch['audio_features'].to(device)
        model(inputs)
    
    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)

# --- 4. RICERCA PERCORSI DATASET ---
def find_dataset_paths():
    """Ricerca i percorsi dei dataset"""
    possible_paths = [
        Path('/kaggle/input/'),
        Path.home() / '.cache' / 'kagglehub' / 'datasets',
        Path('/root/.cache/kagglehub/datasets'),
        Path('/tmp/kagglehub/datasets'),
        Path('./data'),
        Path('../data'),
        Path('../../data'),
    ]
    
    ravdess_path = None
    
    for base_path in possible_paths:
        if base_path.exists():
            for root, dirs, files in os.walk(base_path):
                if 'ravdess-emotional-speech-audio' in dirs and not ravdess_path:
                    ravdess_path = Path(root) / 'ravdess-emotional-speech-audio'
    
    return ravdess_path

# --- 4. FUNZIONE DI TRAINING ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()  
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Training", leave=False)
    
    for batch in loop:
        data = batch['audio_features'].to(device)  # Sposta i dati sulla GPU
        targets = batch['emotion_id'].to(device)   # Sposta le etichette sulla GPU

        # 1. Forward Pass
        scores = model(data)         # Output shape: (Batch, Num_Classes)
        loss = criterion(scores, targets)

        # 2. Backward Pass
        optimizer.zero_grad()        # Pulisci i gradienti vecchi
        loss.backward()              # Calcola i nuovi gradienti
        optimizer.step()             # Aggiorna i pesi

        # 3. Metriche
        running_loss += loss.item()
        _, predictions = scores.max(1) # Prendi l'indice della probabilità più alta
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

        # Aggiorna barra progresso
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy

# --- 5. FUNZIONE DI VALIDATION ---
def validate(model, loader, criterion, device):
    model.eval()  # Imposta modalità valutazione (spegne Dropout)
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # Niente gradienti in validazione (risparmia memoria)
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

    # Ricerca percorsi
    ravdess_path = find_dataset_paths()
    
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

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

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
            "weight_decay": 1e-4,
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
    print(f"Weight Decay (L2):     {1e-4}")
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

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_RAVDESS_dataloader, criterion, optimizer, DEVICE)
        
        # Validation
        val_loss, val_acc = validate(model, val_RAVDESS_dataloader, criterion, DEVICE)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # LOGICA SWA: Dopo SWA_START_EPOCH, inizia averaging dei pesi
        if epoch >= SWA_START_EPOCH:
            if not using_swa:
                print(f"\n🔄 Attivazione SWA da epoch {epoch+1}")
                using_swa = True
            
            # Update SWA model con i pesi correnti
            swa_model.update_parameters(model)
            swa_scheduler.step()
            
            # Valuta anche il swa_model periodicamente
            if (epoch + 1) % 5 == 0:  # Ogni 5 epoche
                print(f"\n📊 Valutazione SWA model...")
                # Update batch normalization statistics
                update_bn_custom(train_RAVDESS_dataloader, swa_model, DEVICE)
                swa_val_loss, swa_val_acc = validate(swa_model, val_RAVDESS_dataloader, criterion, DEVICE)
                print(f"SWA Val Loss: {swa_val_loss:.4f} | SWA Val Acc: {swa_val_acc:.2f}%")
                
                # Log SWA metrics
                wandb.log({
                    "swa_val_loss": swa_val_loss,
                    "swa_val_accuracy": swa_val_acc
                })
                
                # Salva il miglior SWA model
                if swa_val_acc > best_swa_val_acc:
                    best_swa_val_acc = swa_val_acc
                    swa_checkpoint_path = Path(__file__).parent / "checkpoints" / "best_swa_model.pth"
                    save_swa_checkpoint(swa_model, str(swa_checkpoint_path))
                    print(">>> SWA Model Saved!")
        else:
            # Prima di SWA, usa il normale scheduler
            scheduler.step(val_loss)

        # Log metriche su W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        # Checkpoint: Salva il modello se è il migliore finora
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = Path(__file__).parent / "checkpoints" / "best_model.pth"
            torch.save(model.state_dict(), str(checkpoint_path))
            print(">>> Model Saved!")

        # Early Stopping
        early_stopping.step(val_loss)
        if early_stopping.should_stop:
            print(f"\n⏹️ Early stopping attivato dopo {epoch+1} epoche")
            break

    # Valutazione finale del SWA model
    if using_swa:
        print(f"\n🔄 Valutazione finale SWA model...")
        update_bn_custom(train_RAVDESS_dataloader, swa_model, DEVICE)
        final_swa_val_loss, final_swa_val_acc = validate(swa_model, val_RAVDESS_dataloader, criterion, DEVICE)
        print(f"Final SWA Val Loss: {final_swa_val_loss:.4f} | Final SWA Val Acc: {final_swa_val_acc:.2f}%")
        
        # Salva il modello SWA finale
        if final_swa_val_acc > best_swa_val_acc:
            swa_checkpoint_path = Path(__file__).parent / "checkpoints" / "best_swa_model.pth"
            save_swa_checkpoint(swa_model, str(swa_checkpoint_path))
            best_swa_val_acc = final_swa_val_acc
            print(">>> Final SWA Model Saved!")

    print("\n" + "="*80)
    print("✅ Training Complete!")
    print(f"Best Validation Accuracy (Regular): {best_val_acc:.2f}%")
    if using_swa:
        print(f"Best Validation Accuracy (SWA):     {best_swa_val_acc:.2f}%")
        print(f"\n💡 SWA Improvement: {(best_swa_val_acc - best_val_acc):+.2f}%")
        print(f"\n📦 Checkpoints salvati in ./checkpoints/:")
        print(f"   • best_model.pth     → Modello standard")
        print(f"   • best_swa_model.pth → Modello SWA (raccomandato)")
    print("="*80)

    # Chiudi W&B
    wandb.finish()