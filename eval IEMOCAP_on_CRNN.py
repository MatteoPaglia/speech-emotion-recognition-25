import torch
import torch.nn as nn
from pathlib import Path
import os
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataset.custom_iemocap_dataset import CustomIEMOCAPDataset
from models.model import CRNN_BiLSTM
from utils.dataset_utils import find_dataset_in_cache, validate_dataset
from utils.filtered_dataset import FilteredDatasetWrapper

# --- CONFIGURAZIONE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_CLASSES = 4
TIME_STEPS = 200
MEL_BANDS = 128

EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry']
MODEL_PATH = 'checkpoints/best_model_iemocap.pth'

# --- RICERCA PERCORSI DATASET ---
def find_dataset_path():
    """Ricerca il percorso del dataset IEMOCAP nella cache kagglehub"""
    print(f"\n🔍 Ricerca IEMOCAP nella cache kagglehub...")
    
    dataset_path = find_dataset_in_cache('iemocap')
    
    if not dataset_path or not validate_dataset(dataset_path, 'iemocap'):
        raise ValueError(
            f"❌ IEMOCAP non trovato nella cache!\n"
            f"   Percorso cache: {Path.home() / '.cache' / 'kagglehub' / 'datasets'}\n"
            f"   Esegui: python utils/download_dataset.py per scaricare i dataset"
        )
    
    return dataset_path

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
            data = batch['mel_spectrogram'].to(device)
            targets = batch['label'].to(device)
            
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
    print("📊 METRICHE DI VALUTAZIONE - IEMOCAP")
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
def plot_confusion_matrix(predictions, true_labels, save_path="confusion_matrix_iemocap.png"):
    """Plotta e salva la confusion matrix"""
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=EMOTION_LABELS, 
                yticklabels=EMOTION_LABELS,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - IEMOCAP Test Set')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n✅ Confusion matrix salvata: {save_path}")
    plt.show()

# --- MAIN ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"Dataset: IEMOCAP\n")
    
    # 1. Ricerca dataset
    dataset_path = find_dataset_path()
    print(f"✅ IEMOCAP trovato: {dataset_path}\n")
    
    # 2. Carica dataset TEST
    print("Loading IEMOCAP test set...")
    test_dataset_raw = CustomIEMOCAPDataset(dataset_root=str(dataset_path), split='test')
    test_dataset = FilteredDatasetWrapper(test_dataset_raw)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"✅ Test samples: {len(test_dataset)}\n")
    
    # 3. Carica modello
    print("Loading model...")
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"❌ Modello non trovato: {MODEL_PATH}\n   Assicurati di aver eseguito il training prima!")
    
    model = CRNN_BiLSTM(batch_size=BATCH_SIZE, time_steps=TIME_STEPS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    print(f"✅ Modello caricato da {MODEL_PATH}\n")
    
    # 4. Test
    print("="*80)
    print("TESTING IN CORSO - IEMOCAP")
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
