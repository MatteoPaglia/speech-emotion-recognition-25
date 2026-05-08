import torch
import torch.nn as nn 
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
import sys
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
import wandb
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.model_selection import GroupKFold
import argparse

# Aggiungi project root al path per gli import
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as Config
from dataset.custom_iemocap_dataset import CustomIEMOCAPDataset
from models import get_model
from utils.training_utils import SimpleEarlyStopping, save_swa_checkpoint, update_bn_custom

#Choose model for training
MODEL_TYPE = 'CRNN_BiLSTM' # Options: 'CRNN_BiLSTM', 'CRNN_BiGRU'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Crea cartella checkpoints se non esiste
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "iemocap_only"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Importa da config
BATCH_SIZE = Config.BATCH_SIZE_IEMOCAP
LEARNING_RATE = Config.LEARNING_RATE_IEMOCAP
NUM_EPOCHS = Config.NUM_EPOCHS_IEMOCAP
NUM_CLASSES = Config.NUM_CLASSES_IEMOCAP
TIME_STEPS = Config.TIME_STEPS_IEMOCAP
MEL_BANDS = Config.MEL_BANDS_IEMOCAP
DROPOUT = Config.DROPOUT_IEMOCAP
SPEC_FREQ_MASK = Config.SPEC_FREQ_MASK_IEMOCAP
SPEC_TIME_MASK = Config.SPEC_TIME_MASK_IEMOCAP
CLASS_WEIGHTS = Config.CLASS_WEIGHTS_IEMOCAP
SWA_START_EPOCH = Config.SWA_START_EPOCH_IEMOCAP
SWA_LR = Config.SWA_LR_IEMOCAP
WEIGHT_DECAY = Config.WEIGHT_DECAY_IEMOCAP

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

    # LOGICA DI SELEZIONE DEL PATH
    if DATASET_PATH_ARG:
        iemocap_path = Path(DATASET_PATH_ARG)
    else:
        if Config.COLAB_IEMOCAP_PATH and Config.COLAB_IEMOCAP_PATH.exists():
            iemocap_path = Config.COLAB_IEMOCAP_PATH
        else:
            iemocap_path = Path(Config.IEMOCAP_PATH)
    
    if not iemocap_path or not iemocap_path.exists():
        raise ValueError(f"❌ IEMOCAP non trovato in: {iemocap_path}! Verifica il percorso.")
    
    print(f"\n✅ IEMOCAP path impostato su: {iemocap_path}\n")
    
    # SCAN DEI FILE E DEGLI SPEAKER
    all_files, speaker_ids = CustomIEMOCAPDataset.get_all_speakers(iemocap_path)
    if not all_files:
        raise ValueError("Nessun file trovato durante lo scan del dataset IEMOCAP.")
    
    print(f"Trovati {len(all_files)} file validi appartenenti a {len(set(speaker_ids))} speaker.")

    # Genera un ID del gruppo per W&B CV
    timestamp = (datetime.now() + timedelta(hours=1)).strftime("%Y%m%d_%H%M%S")
    cv_group_id = f"cv-iemocap-{timestamp}"
    
    fold_results_reg = []
    fold_results_swa = []

    # Istanzia iteratore GroupKFold
    gkf = GroupKFold(n_splits=N_SPLITS)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(all_files, groups=speaker_ids)):
        fold_num = fold + 1
        print("\n" + "="*80)
        print(f"🚀 INIZIO FOLD {fold_num}/{N_SPLITS}")
        print("="*80)

        # Ricava gli ID univoci degli speaker di train e val per questo fold
        train_speakers = set([speaker_ids[idx] for idx in train_idx])
        val_speakers = set([speaker_ids[idx] for idx in val_idx])
        
        print(f"Speakers Train ({len(train_speakers)}): {sorted(list(train_speakers))}")
        print(f"Speakers Val ({len(val_speakers)}): {sorted(list(val_speakers))}")

        # Istanzia i dataset usando il parametro `allowed_speakers`
        train_dataset = CustomIEMOCAPDataset(
            dataset_root=str(iemocap_path),
            allowed_speakers=train_speakers,
            is_train=True,
            spec_freq_mask=SPEC_FREQ_MASK,
            spec_time_mask=SPEC_TIME_MASK
        )
        val_dataset = CustomIEMOCAPDataset(
            dataset_root=str(iemocap_path),
            allowed_speakers=val_speakers,
            is_train=False
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        # RE-INIZIALIZZA MODELLO E OTTIMIZZATORE (IMPORTANTE PER NON LEAKARE PARAMS)
        model = get_model(MODEL_TYPE, batch_size=BATCH_SIZE, time_steps=TIME_STEPS, dropout=DROPOUT, channel=1).to(DEVICE)
        
        class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)
        class_weights = class_weights / class_weights.sum()
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)

        # INIZIALIZZA WANDB per il Fold
        run_name = f"fold_{fold_num}_{MODEL_TYPE}"
        wandb.init(
            project="speech-emotion-recognition",
            group=cv_group_id,
            name=run_name,
            reinit=True,
            config={
                "fold": fold_num,
                "train_speakers": sorted(list(train_speakers)),
                "val_speakers": sorted(list(val_speakers)),
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "epochs": NUM_EPOCHS,
                "num_classes": NUM_CLASSES,
                "time_steps": TIME_STEPS,
                "mel_bands": MEL_BANDS,
                "architecture": MODEL_TYPE,
                "dataset": "IEMOCAP_CV",
                "optimizer": "Adam",
                "weight_decay": WEIGHT_DECAY,
                "loss_function": "CrossEntropyLoss",
                "swa_start_epoch": SWA_START_EPOCH,
                "swa_lr": SWA_LR,
                "dropout": DROPOUT,
                "spec_freq_mask": SPEC_FREQ_MASK,
                "spec_time_mask": SPEC_TIME_MASK,
                "class_weights": CLASS_WEIGHTS
            }
        )

        best_val_acc = 0.0
        best_swa_val_acc = 0.0
        using_swa = False

        print(f"\nTraining...")
        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

            epoch_marker = ""
            if val_acc > best_val_acc:
                epoch_marker = "⭐"
                best_val_acc = val_acc
                torch.save(model.state_dict(), str(CHECKPOINT_DIR / f"best_model_fold_{fold_num}.pth"))
            
            print(f"Ep [{epoch+1:<2}/{NUM_EPOCHS}] T_loss: {train_loss:.3f} | T_acc: {train_acc:.1f}% | V_loss: {val_loss:.3f} | V_acc: {val_acc:.1f}% {epoch_marker}")

            if epoch >= SWA_START_EPOCH:
                if not using_swa:
                    using_swa = True
                
                swa_model.update_parameters(model)
                swa_scheduler.step()
                
                if (epoch + 1) % 5 == 0:
                    update_bn_custom(train_loader, swa_model, DEVICE)
                    swa_val_loss, swa_val_acc = validate(swa_model, val_loader, criterion, DEVICE)
                    
                    wandb.log({
                        "fold": fold_num,
                        "epoch": epoch + 1,
                        "swa_val_loss": swa_val_loss,
                        "swa_val_accuracy": swa_val_acc
                    })
                    
                    if swa_val_acc > best_swa_val_acc:
                        best_swa_val_acc = swa_val_acc
                        save_swa_checkpoint(swa_model, str(CHECKPOINT_DIR / f"best_swa_model_fold_{fold_num}.pth"))
            else:
                scheduler.step(val_loss)

            wandb.log({
                "fold": fold_num,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
            
        if using_swa:
            update_bn_custom(train_loader, swa_model, DEVICE)
            _, final_swa_val_acc = validate(swa_model, val_loader, criterion, DEVICE)
            if final_swa_val_acc > best_swa_val_acc:
                best_swa_val_acc = final_swa_val_acc
        
        fold_results_reg.append(best_val_acc)
        fold_results_swa.append(best_swa_val_acc)
        
        print(f"\n✅ Fine Fold {fold_num} | Best Val Acc: {best_val_acc:.2f}% | Best SWA Acc: {best_swa_val_acc:.2f}%")
        wandb.finish()

    reg_mean, reg_std = np.mean(fold_results_reg), np.std(fold_results_reg)
    swa_mean, swa_std = np.mean(fold_results_swa), np.std(fold_results_swa)

    print("\n" + "="*80)
    print(f"🏆 RISULTATI CROSS-VALIDATION IEMOCAP ({N_SPLITS} Folds)")
    print("="*80)
    for i in range(N_SPLITS):
        print(f"Fold {i+1}: Regular Acc = {fold_results_reg[i]:.2f}%, SWA Acc = {fold_results_swa[i]:.2f}%")
    print("-" * 80)
    print(f"Metrics (Regular): MEAN = {reg_mean:.2f}%, STD = {reg_std:.2f}%")
    print(f"Metrics (SWA):     MEAN = {swa_mean:.2f}%, STD = {swa_std:.2f}%")
    print("="*80 + "\n")
