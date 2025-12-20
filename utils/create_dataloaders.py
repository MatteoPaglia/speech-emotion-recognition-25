from torch.utils.data import DataLoader
from dataset.custom_ravdess_dataset import CustomRAVDESSDataset
from torch.nn.utils.rnn import pad_sequence


def collate_fn_unified(batch):
    """
    Collate function unificata per RAVDESS e IEMOCAP.
    
    Funziona con qualsiasi dataset che ritorna:
        - 'audio': waveform [1, num_samples]
        - 'emotion_id': int (0-3)
    
    Args:
        batch: List di dict da __getitem__
    
    Returns:
        dict con audio paddati e labels
    """
    # Estrai waveforms [1, samples] -> [samples]
    audios = [item['audio'].squeeze(0) for item in batch]
    
    # Estrai emotion IDs
    emotion_ids = torch.tensor([item['emotion_id'] for item in batch], dtype=torch.long)
    
    # Salva lunghezze originali
    audio_lengths = torch.tensor([audio.shape[0] for audio in audios], dtype=torch.long)
    
    # Padding: [batch_size, max_length]
    audios_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)
    
    # Aggiungi dimensione canale: [batch_size, 1, max_length]
    audios_padded = audios_padded.unsqueeze(1)
    
    return {
        'audio': audios_padded,           # [batch_size, 1, max_length]
        'emotion_id': emotion_ids,        # [batch_size]
        'audio_lengths': audio_lengths    # [batch_size]
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