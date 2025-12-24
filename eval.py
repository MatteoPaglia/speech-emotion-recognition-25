import torch
import torch.nn as nn
from pathlib import Path
import os
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataset.custom_ravdess_dataset import CustomRAVDESSDataset
from models.model import CRNN_BiLSTM

# --- CONFIGURAZIONE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_CLASSES = 4
TIME_STEPS = 200
MEL_BANDS = 128
MODEL_PATH = "checkpoints/best_model.pth"

EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry']

# --- RICERCA PERCORSI DATASET ---
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

# --- FUNZIONE DI TESTING ---
def test_model(model, loader, device):
    """
    Testa il modello su un dataset
    
    Returns:
        predictions: liste di predizioni
        true_labels: liste di label vere
        all_probs: probabilità per ogni classe
    """
    model.eval()
    predictions = []
    true_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            data = batch['audio_features'].to(device)
            targets = batch['emotion_id'].to(device)
            
            scores = model(data)
            probs = torch.softmax(scores, dim=1)
            
            _, preds = scores.max(1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(predictions), np.array(true_labels), np.array(all_probs)

# --- FUNZIONE PER STAMPARE METRICHE ---
def print_metrics(predictions, true_labels):
    """Stampa metriche di valutazione dettagliate"""
    from sklearn.metrics import f1_score
    
    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
    weighted_f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    
    print("\n" + "="*80)
    print("📊 METRICHE DI VALUTAZIONE - FASE 1 (RAVDESS BASELINE)")
    print("="*80)
    
    # Metriche principali
    print(f"\n🎯 METRICHE PRINCIPALI:")
    print(f"   ✅ Accuracy:           {accuracy*100:.2f}%")
    print(f"   📈 Macro-Avg F1:       {macro_f1:.4f}")
    print(f"   📊 Weighted-Avg F1:    {weighted_f1:.4f}")
    
    # Class distribution
    unique, counts = np.unique(true_labels, return_counts=True)
    print(f"\n📋 CLASS DISTRIBUTION (Test Set):")
    for idx, count in zip(unique, counts):
        pct = (count / len(true_labels)) * 100
        print(f"   {EMOTION_LABELS[idx]:10s}: {count:3d} samples ({pct:5.1f}%)")
    
    print("\n🎭 DETAILED CLASSIFICATION REPORT:")
    print(classification_report(true_labels, predictions, target_names=EMOTION_LABELS))
    
    return accuracy, macro_f1, weighted_f1

# --- FUNZIONE PER CONFUSION MATRIX ---
def plot_confusion_matrix(predictions, true_labels, save_path="confusion_matrix.png"):
    """Plotta e salva la confusion matrix"""
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=EMOTION_LABELS, 
                yticklabels=EMOTION_LABELS,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - RAVDESS Test Set')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n✅ Confusion matrix salvata: {save_path}")
    plt.show()

# --- MAIN ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}\n")
    
    # 1. Ricerca dataset
    ravdess_path = find_dataset_paths()
    if not ravdess_path or not ravdess_path.exists():
        raise ValueError("❌ RAVDESS non trovato!")
    
    print(f"✅ RAVDESS trovato: {ravdess_path}\n")
    
    # 2. Carica dataset TEST
    print("Loading RAVDESS test set...")
    test_dataset = CustomRAVDESSDataset(dataset_root=str(ravdess_path), split='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"✅ Test samples: {len(test_dataset)}\n")
    
    # 3. Carica modello
    print("Loading model...")
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"❌ Modello non trovato: {MODEL_PATH}")
    
    model = CRNN_BiLSTM(batch_size=BATCH_SIZE, time_steps=TIME_STEPS, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    print(f"✅ Modello caricato da {MODEL_PATH}\n")
    
    # 4. Test
    print("="*80)
    print("TESTING IN CORSO...")
    print("="*80)
    predictions, true_labels, all_probs = test_model(model, test_loader, DEVICE)
    
    # 5. Metriche
    accuracy, macro_f1, weighted_f1 = print_metrics(predictions, true_labels)
    
    # 6. Confusion Matrix
    plot_confusion_matrix(predictions, true_labels)
    
    print("\n" + "="*80)
    print("✅ Evaluation Complete!")
    print(f"   Final Accuracy: {accuracy*100:.2f}%")
    print(f"   Macro-Avg F1:   {macro_f1:.4f}")
    print(f"   Weighted-Avg F1: {weighted_f1:.4f}")
    print("="*80)
