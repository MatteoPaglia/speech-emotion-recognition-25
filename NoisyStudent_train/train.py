import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import os
import sys
import json
from torch.utils.data import DataLoader, ConcatDataset
from datetime import datetime, timedelta
import wandb
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import argparse

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as Config
from dataset.custom_ravdess_dataset import CustomRAVDESSDataset
from dataset.custom_iemocap_dataset import CustomIEMOCAPDataset
from models import get_model
from utils.training_utils import save_swa_checkpoint, update_bn_custom

# --- 1. ARGPARSE ---
parser = argparse.ArgumentParser(description='Train Noisy Student Speech Emotion Recognition Model')
parser.add_argument('--model', type=str, default='CRNN_BiLSTM', 
                    choices=['CRNN_BiLSTM', 'CRNN_BiGRU'],
                    help='Model type to use (default: CRNN_BiLSTM)')
parser.add_argument('--ravdess_path', type=str, default=None, 
                    help='Absolute or relative path of RAVDESS dataset')
parser.add_argument('--iemocap_path', type=str, default=None, 
                    help='Absolute or relative path of IEMOCAP dataset')
args = parser.parse_args()

MODEL_TYPE = args.model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. CONFIGURATION ---
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "noisy_student"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = Config.BATCH_SIZE_STUDENT_IEMOCAP
LEARNING_RATE = Config.LEARNING_RATE_STUDENT_IEMOCAP
NUM_EPOCHS = Config.NUM_EPOCHS_STUDENT_IEMOCAP
TIME_STEPS = Config.TIME_STEPS_RAVDESS  # RAVDESS and IEMOCAP share the same length natively (3s)
DROPOUT = Config.DROPOUT_STUDENT_IEMOCAP
WEIGHT_DECAY = Config.WEIGHT_DECAY_STUDENT_IEMOCAP

SWA_START_EPOCH = Config.SWA_START_EPOCH_STUDENT
SWA_LR = Config.SWA_LR_STUDENT

# Noisy parameters
ADDITIVE_NOISE_SNR = (Config.ADDITIVE_NOISE_SNR_MIN, Config.ADDITIVE_NOISE_SNR_MAX)


# --- 3. TRAINING FUNCTIONS ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()  
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in loader:
        data = batch['audio_features'].to(device)
        
        # Determine targets. The ConcatDataset will return `emotion_id` 
        # For IEMOCAP, if it has a `pseudo_emotion_id`, we use it. Else we use `emotion_id`.
        # However, CustomIEMOCAPDataset already maps pseudo_label inside `pseudo_emotion_id`
        # Let's cleanly fetch it.
        
        if 'pseudo_emotion_id' in batch:
            targets = batch['pseudo_emotion_id'].to(device)
        else:
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

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            data = batch['audio_features'].to(device)
            targets = batch['emotion_id'].to(device) # Validation is always Ground Truth

            scores = model(data)
            loss = criterion(scores, targets)

            running_loss += loss.item()
            _, predictions = scores.max(1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy

# --- 4. MAIN LOOP ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # PATH RESOLUTION
    ravdess_path = Path(args.ravdess_path) if args.ravdess_path else PROJECT_ROOT / Config.RAVDESS_PATH
    iemocap_path = Path(args.iemocap_path) if args.iemocap_path else PROJECT_ROOT / Config.IEMOCAP_PATH
    
    if not ravdess_path.exists(): raise ValueError(f"❌ RAVDESS not found at: {ravdess_path}")
    if not iemocap_path.exists(): raise ValueError(f"❌ IEMOCAP not found at: {iemocap_path}")

    # PSEUDO LABELS LOADING
    pseudo_labels_path = PROJECT_ROOT / "NoisyStudent_train" / "pseudo_labels_iemocap.json"
    if not pseudo_labels_path.exists():
        raise FileNotFoundError(f"❌ Pseudo-labels {pseudo_labels_path} not found. Run generate_pseudo_labels.py first!")
        
    with open(pseudo_labels_path, 'r') as f:
        pseudo_labels_dict = json.load(f)
        
    # DATASETS
    # Ravdess (Labeled Source) - 100% Train
    train_ravdess = CustomRAVDESSDataset(
        dataset_root=str(ravdess_path),
        allowed_speakers=None, # Use all 24 speakers to maximize labelled data
        is_train=True,
        spec_freq_mask=Config.SPEC_FREQ_MASK_IEMOCAP, 
        spec_time_mask=Config.SPEC_TIME_MASK_IEMOCAP,
        add_gaussian_noise_snr=ADDITIVE_NOISE_SNR
    )
    
    # IEMOCAP Train (Pseudo-labeled Target) - Session 1, 2, 3
    train_iemocap = CustomIEMOCAPDataset(
        dataset_root=str(iemocap_path),
        allowed_speakers={'1F','1M','2F','2M','3F','3M'},
        is_train=True,
        spec_freq_mask=Config.SPEC_FREQ_MASK_IEMOCAP,
        spec_time_mask=Config.SPEC_TIME_MASK_IEMOCAP,
        pseudo_labels_dict=pseudo_labels_dict, # Assign pseudo labels
        add_gaussian_noise_snr=ADDITIVE_NOISE_SNR
    )
    
    # Combine datasets
    train_combined = ConcatDataset([train_ravdess, train_iemocap])
    train_loader = DataLoader(train_combined, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # IEMOCAP Validation - Session 4
    val_iemocap = CustomIEMOCAPDataset(
        dataset_root=str(iemocap_path),
        allowed_speakers={'4F','4M'},
        is_train=False, # Clean evaluation
    )
    val_loader = DataLoader(val_iemocap, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Total Student Training Samples: {len(train_combined)} (Ravdess: {len(train_ravdess)}, IEMOCAP Pseudo: {len(train_iemocap)})")
    print(f"Total Student Validation Samples: {len(val_iemocap)}")

    # W&B INIT
    timestamp = (datetime.now() + timedelta(hours=1)).strftime("%Y%m%d_%H%M%S")
    run_name = f"noisy_student_{MODEL_TYPE}_{timestamp}"
    wandb.init(
        project="speech-emotion-recognition",
        group="noisy-student",
        name=run_name,
        config={
            "model": MODEL_TYPE,
            "pseudo_labels": len(pseudo_labels_dict),
            "snr_noise": ADDITIVE_NOISE_SNR,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY
        }
    )

    # MODEL & OPTIMIZER
    model = get_model(MODEL_TYPE, batch_size=BATCH_SIZE, time_steps=TIME_STEPS, dropout=DROPOUT, channel=1).to(DEVICE)
    
    # Simple cross-entropy 
    class_weights = torch.tensor(Config.CLASS_WEIGHTS_IEMOCAP, dtype=torch.float32).to(DEVICE)
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=Config.LABEL_SMOOTHING)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)

    best_val_loss = float('inf')
    patience_counter = 0

    print("\n🚀 STARTING NOISY STUDENT TRAINING LOOP")
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n[Epoch {epoch}/{NUM_EPOCHS}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        print(f"📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        print(f"📉 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Scheduling
        if epoch >= SWA_START_EPOCH:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            print("🔄 SWA Model updated")
        else:
            scheduler.step(val_loss)

        # Logging
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, CHECKPOINT_DIR / "best_model.pth")
            print("⭐ Best model saved!")
        else:
            patience_counter += 1
            print(f"⚠️ Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE_STUDENT}")

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE_STUDENT:
            print(f"🛑 Early stopping reached (Patience >= {Config.EARLY_STOPPING_PATIENCE_STUDENT}), ma è stato DISATTIVATO. Training in corso...")
            # break

    # FINAL SWA UPDATE
    if epoch >= SWA_START_EPOCH:
        print("\n🏁 Finalizing SWA Model...")
        update_bn_custom(train_loader, swa_model, device=DEVICE)
        swa_val_loss, swa_val_acc = validate(swa_model, val_loader, criterion, DEVICE)
        
        save_swa_checkpoint(swa_model, str(CHECKPOINT_DIR / "best_swa_model.pth"))
        print(f"✅ Best SWA Model saved! Val Loss: {swa_val_loss:.4f} | Val Acc: {swa_val_acc:.2f}%")
        
        wandb.log({
            "swa_val_loss": swa_val_loss,
            "swa_val_acc": swa_val_acc
        })

    wandb.finish()
    print("✅ Training Complete")
