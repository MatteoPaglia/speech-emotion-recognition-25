from torch.utils.data import DataLoader
from dataset.custom_ravdess_dataset import CustomRAVDESSDataset
from torch.nn.utils.rnn import pad_sequence


def collate_fn_unified(batch):
    """
    Collate function ottimizzata per Spettrogrammi a lunghezza FISSA.
    Non fa padding dinamico perché i dati sono già uniformati nel Dataset (es. 3s).
    """
    # 1. Stack delle features (Spettrogrammi)
    # Poiché sono tutti della stessa dimensione (es. 3s), usiamo stack invece di pad_sequence
    # Input: Lista di [1, n_mels, time] -> Output: [Batch, 1, n_mels, time]
    # Nota: Assicurati che il Dataset ritorni la chiave 'audio_features'
    audio_features = torch.stack([item['audio_features'] for item in batch])
    
    # 2. Stack delle labels
    emotion_ids = torch.tensor([item['emotion_id'] for item in batch], dtype=torch.long)
    
    return {
        'audio_features': audio_features,  # [B, 1, 128, T] -> Pronto per la CNN
        'emotion_id': emotion_ids          # [B]
    }

def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=32,
    num_workers=1,
    pin_memory=True,
    device='cuda',
    dataset_name='Dataset'
):
    """
    Funzione UNIFICATA per creare DataLoaders.
    Funziona sia per RAVDESS che IEMOCAP.
    """
    if device == 'cpu':
        num_workers = 0
        pin_memory = False
    
    print("=" * 80)
    print(f"📦 CREAZIONE DATALOADERS - {dataset_name}")
    print("=" * 80)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn_unified  
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn_unified  
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn_unified
    )
    
    print(f"\n✅ DataLoaders creati:")
    print(f"   Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"   Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"   Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    print("=" * 80)
    
    return train_loader, val_loader, test_loader