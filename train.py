import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
from torch.utils.data import DataLoader

from dataset.custom_ravdess_dataset import CustomRAVDESSDataset
from models.model import CRNN_BiLSTM 

# --- 2. CONFIGURAZIONE (IPERPARAMETRI) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Se usi Mac M1/M2 puoi usare: torch.device("mps")

# Crea cartella checkpoints se non esiste
Path("checkpoints").mkdir(exist_ok=True)

BATCH_SIZE = 32
LEARNING_RATE = 0.0001  # Adam lavora bene con 1e-3 o 1e-4
NUM_EPOCHS = 100
NUM_CLASSES = 4       # 8 We consider only 4 emotions: Neutral, Happy, Sad, Angry
TIME_STEPS = 200      # Consider avg or max time steps calculated before 
MEL_BANDS = 128

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

# --- 3. RICERCA PERCORSI DATASET ---
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
    train_RAVDESS_dataset = CustomRAVDESSDataset(dataset_root=str(ravdess_path), split='train')
    val_RAVDESS_dataset = CustomRAVDESSDataset(dataset_root=str(ravdess_path), split='validation')
    
    print(f"Train samples: {len(train_RAVDESS_dataset)}")
    print(f"Val samples: {len(val_RAVDESS_dataset)}")

    # Create RAVDESS DataLoaders
    
    train_RAVDESS_dataloader = DataLoader(train_RAVDESS_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_RAVDESS_dataloader = DataLoader(val_RAVDESS_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Inizializzazione Modello
    model = CRNN_BiLSTM(batch_size=BATCH_SIZE, time_steps=TIME_STEPS).to(DEVICE)
    
    # Stampa dell'architettura del modello
    print("\n" + "="*80)
    print("🏗️ ARCHITETTURA DEL MODELLO")
    print("="*80)
    print(model)
    print("="*80 + "\n")

    # Class weights per bilanciare le classi (normalizzati)
    # Ordine delle classi (da EMOTION_ID_MAP in custom_ravdess_dataset.py):
    # 0: Neutral (0.8) | 1: Happy (1.2) | 2: Sad (1.2) | 3: Angry (1.2)
    class_weights = torch.tensor([0.8, 1.2, 1.2, 1.2], dtype=torch.float32).to(DEVICE)
    # Normalizza i pesi (somma = 1)
    class_weights = class_weights / class_weights.sum()

    criterion = nn.CrossEntropyLoss(weight=class_weights,label_smoothing=0.1)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    # Ciclo delle Epoche
    best_val_acc = 0.0
    early_stopping = SimpleEarlyStopping(patience=10)

    # --- STAMPA IPERPARAMETRI ---
    print("\n" + "="*80)
    print("🔧 IPERPARAMETRI DI TRAINING")
    print("="*80)
    print(f"Device:                {DEVICE}")
    print(f"Batch Size:            {BATCH_SIZE}")
    print(f"Learning Rate:         {LEARNING_RATE}")
    print(f"Weight Decay (L2):     {1e-3}")
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

    print("\n" + "="*80)
    print("✅ Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("="*80)