import os
import sys
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# Setup project root for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as Config
from models import get_model
from dataset.custom_iemocap_dataset import CustomIEMOCAPDataset

def generate_pseudo_labels():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 1. Load Teacher Model (RAVDESS best model)
    # The teacher should be the best trained model on RAVDESS
    teacher_checkpoint = PROJECT_ROOT / "checkpoints" / "ravdess" / "best_swa_model.pth" # Fallback to best_model if swa missing
    if not teacher_checkpoint.exists():
        teacher_checkpoint = PROJECT_ROOT / "checkpoints" / "ravdess" / "best_model.pth"
        
    if not teacher_checkpoint.exists():
        raise FileNotFoundError(f"❌ Teacher checkpoint not found at {teacher_checkpoint}")

    print(f"Loading Teacher model from: {teacher_checkpoint}")
    
    # Initialize the architecture 
    # Must match the architecture trained on RAVDESS. Assuming CRNN_BiLSTM
    teacher = get_model('CRNN_BiLSTM', 
                        batch_size=Config.BATCH_SIZE_RAVDESS, 
                        time_steps=Config.TIME_STEPS_RAVDESS, 
                        dropout=Config.DROPOUT_RAVDESS, 
                        channel=1)
                        
    state_dict = torch.load(teacher_checkpoint, map_location=DEVICE)
    if 'model_state_dict' in state_dict:
        teacher.load_state_dict(state_dict['model_state_dict'])
    else:
         teacher.load_state_dict(state_dict)
         
    teacher.to(DEVICE)
    teacher.eval()

    # 2. Load IEMOCAP Dataset (Unlabeled split)
    # We want Session 1, 2, and 3 for Unlabeled Training.
    unlabeled_speakers = {'1F','1M','2F','2M','3F','3M'}
    
    print("\nLoading IEMOCAP Dataset (Sessions 1, 2, 3) for pseudo-labeling...")
    iemocap_dataset = CustomIEMOCAPDataset(
        dataset_root=Config.IEMOCAP_PATH,
        allowed_speakers=unlabeled_speakers,
        is_train=False, # Distillation inference should not use data augmentation
        target_length=3.0,
        target_sample_rate=16000,
        target_n_fft=1024,
        target_hop_length=256,
        target_n_mels=128
    )

    loader = DataLoader(iemocap_dataset, batch_size=Config.BATCH_SIZE_IEMOCAP, shuffle=False, num_workers=4)

    # 3. Generate Pseudo-Labels
    print("\nGenerating Pseudo-Labels...")
    teacher_pseudo_labels = {}
    total_samples = 0
    accepted_samples = 0
    
    # We use hard labels with a confidence threshold
    confidence_threshold = getattr(Config, 'CONFIDENCE_THRESHOLD', 0.7)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Teacher Inference"):
            audio_features = batch['audio_features'].to(DEVICE)
            sample_ids = batch['sample_id'] 

            # Forward pass
            logits = teacher(audio_features)
            
            # Use softmax (Temperature = 1.0 for hard labels)
            probs = torch.softmax(logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            
            for i, sample_id in enumerate(sample_ids):
                total_samples += 1
                conf = max_probs[i].item()
                if conf >= confidence_threshold:
                    accepted_samples += 1
                    teacher_pseudo_labels[sample_id] = preds[i].item()

    print(f"\n✅ Total pseudo-labels generated: {total_samples}")
    print(f"🎯 Accepted (>= {confidence_threshold} confidence): {accepted_samples} ({(accepted_samples/total_samples)*100:.2f}%)")
    print(f"🗑️ Discarded: {total_samples - accepted_samples}")

    # 4. Save to JSON
    output_path = PROJECT_ROOT / "NoisyStudent_train" / "pseudo_labels_iemocap.json"
    with open(output_path, 'w') as f:
        json.dump(teacher_pseudo_labels, f, indent=4)
    print(f"\n💾 Saved accepted pseudo-labels to {output_path}")

if __name__ == "__main__":
    generate_pseudo_labels()