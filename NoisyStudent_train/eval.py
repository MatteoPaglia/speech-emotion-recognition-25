import torch
import numpy as np
import sys
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from tabulate import tabulate

# Add project root to sys.path 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as Config
from dataset.custom_iemocap_dataset import CustomIEMOCAPDataset
from dataset.custom_ravdess_dataset import CustomRAVDESSDataset
from models import get_model

parser = argparse.ArgumentParser(description='Evaluate Student vs Teacher Models on Target Domain (IEMOCAP Session 5)')
parser.add_argument('--model', type=str, default='CRNN_BiLSTM', choices=['CRNN_BiLSTM', 'CRNN_BiGRU'],
                    help='Model type used')
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = args.model

# Labels for plots and reports
EMOTIONS = ['neutral', 'happy', 'sad', 'angry']

def evaluate_model(model, loader, desc="Evaluation"):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            data = batch['audio_features'].to(DEVICE)
            targets = batch['emotion_id']
            
            scores = model(data)
            _, preds = scores.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    return np.array(all_targets), np.array(all_preds)

def make_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    save_path = Path(__file__).parent / filename
    plt.savefig(save_path)
    plt.close()
    print(f"  --> Saved Confusion Matrix to {save_path.name}")

if __name__ == '__main__':
    print("="*80)
    print("🎓 NOISY STUDENT / DOMAIN ADAPTATION EVALUATION 🎓")
    print("="*80)
    
    # --- 1. Load TEST Data (Session 5) ---
    print("\nLoading IEMOCAP Session 5 (Unseen Test Set)...")
    test_iemocap = CustomIEMOCAPDataset(
        dataset_root=str(PROJECT_ROOT / Config.IEMOCAP_PATH),
        allowed_speakers={'5F','5M'}, # Session 5 ONLY
        is_train=False
    )
    test_loader = DataLoader(test_iemocap, batch_size=Config.BATCH_SIZE_IEMOCAP, shuffle=False)
    
    # --- 2. Evaluate Baseline (Teacher) ---
    print("\n[Phase 1] Evaluating Baseline (Teacher trained on RAVDESS)")
    teacher_weights_path = PROJECT_ROOT / "checkpoints" / "ravdess" / "best_swa_model.pth"
    if not teacher_weights_path.exists(): teacher_weights_path = PROJECT_ROOT / "checkpoints" / "ravdess" / "best_model.pth"
    
    teacher = get_model(MODEL_TYPE, batch_size=Config.BATCH_SIZE_IEMOCAP, time_steps=Config.TIME_STEPS_RAVDESS, dropout=0.0).to(DEVICE)
    try:
        ts = torch.load(teacher_weights_path, map_location=DEVICE)
        teacher.load_state_dict(ts.get('model_state_dict', ts))
    except Exception as e:
        print(f"Failed to load Teacher: {e}")
        sys.exit(1)
        
    y_true_tgt, y_pred_teacher = evaluate_model(teacher, test_loader)
    teacher_acc = accuracy_score(y_true_tgt, y_pred_teacher)
    teacher_f1 = f1_score(y_true_tgt, y_pred_teacher, average='macro')
    
    print("\nBaseline Results on Spontaneous Speech (Target Domain):")
    print(classification_report(y_true_tgt, y_pred_teacher, target_names=EMOTIONS))
    make_confusion_matrix(y_true_tgt, y_pred_teacher, "Baseline Model on IEMOCAP Session 5", "cm_cross_domain_baseline.png")
    
    
    # --- 3. Evaluate Student (Noisy Student) ---
    print("\n[Phase 2] Evaluating Student (Trained via Noisy Student)")
    student_weights_path = PROJECT_ROOT / "checkpoints" / "noisy_student" / "best_swa_model.pth"
    if not student_weights_path.exists(): student_weights_path = PROJECT_ROOT / "checkpoints" / "noisy_student" / "best_model.pth"
    
    student = get_model(MODEL_TYPE, batch_size=Config.BATCH_SIZE_IEMOCAP, time_steps=Config.TIME_STEPS_RAVDESS, dropout=0.0).to(DEVICE)
    try:
        ss = torch.load(student_weights_path, map_location=DEVICE)
        student.load_state_dict(ss.get('model_state_dict', ss))
    except Exception as e:
        print(f"Failed to load Student: {e}")
        print("Note: Train the student model first!")
        sys.exit(1)
        
    y_true_tgt, y_pred_student = evaluate_model(student, test_loader)
    student_acc = accuracy_score(y_true_tgt, y_pred_student)
    student_f1 = f1_score(y_true_tgt, y_pred_student, average='macro')
    
    print("\nStudent Results on Spontaneous Speech (Target Domain):")
    print(classification_report(y_true_tgt, y_pred_student, target_names=EMOTIONS))
    make_confusion_matrix(y_true_tgt, y_pred_student, "Student Model on IEMOCAP Session 5", "cm_cross_domain_student.png")


    # --- 4. Domain Gap & Recovery Analysis ---
    print("\n" + "="*80)
    print("📊 ROBUSTNESS & GAP RECOVERY METRICS")
    print("="*80)
    
    table = [
        ["Baseline (Teacher)", f"{teacher_acc*100:.2f}%", f"{teacher_f1*100:.2f}%"],
        ["Noisy Student", f"{student_acc*100:.2f}%", f"{student_f1*100:.2f}%"],
        ["Absolute Improvement", f"{(student_acc-teacher_acc)*100:+.2f}%", f"{(student_f1-teacher_f1)*100:+.2f}%"]
    ]
    print(tabulate(table, headers=["Model", "Test Accuracy", "Macro F1-Score"], tablefmt="pretty"))
    
    print("\nSummary:")
    print("The 'Absolute Improvement' tells us how much of the domain shift gap (Sim-to-Real)")
    print("was successfully mitigated using only Unlabeled Target data via Noisy Student.\n")
    print("Also check the confusion matrices in the NoisyStudent_train folder to analyze Arousal-Valence confusions.")
