"""
Dataset wrapper che filtra i campioni corrotti/mancanti al caricamento.
Evita che il DataLoader riceva None durante l'iterazione.
"""

import torch
from torch.utils.data import Dataset


class FilteredDatasetWrapper(Dataset):
    """
    Wrapper che filtra i campioni corrotti dal dataset originale.
    Carica tutti i campioni al primo accesso e memorizza gli indici validi.
    """
    
    def __init__(self, dataset):
        """
        Args:
            dataset: Dataset PyTorch che potrebbe ritornare None
        """
        self.dataset = dataset
        self.valid_indices = []
        
        print(f"\n🔍 Scansione dataset per identificare campioni validi...")
        print(f"   Totale campioni: {len(dataset)}")
        
        # Scansiona il dataset e identifica gli indici validi
        failed_count = 0
        for idx in range(len(dataset)):
            try:
                sample = self.dataset[idx]
                if sample is not None:
                    self.valid_indices.append(idx)
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:  # Stampa solo i primi 5 errori
                    print(f"   ⚠️  Indice {idx}: {str(e)[:50]}")
        
        print(f"✅ Campioni validi: {len(self.valid_indices)}/{len(dataset)}")
        if failed_count > 0:
            print(f"❌ Campioni filtrati (corrotti/mancanti): {failed_count}")
    
    def __len__(self):
        """Ritorna il numero di campioni validi"""
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        Ritorna il campione valido all'indice idx.
        Garantisce di non ritornare mai None.
        """
        valid_idx = self.valid_indices[idx]
        sample = self.dataset[valid_idx]
        
        # Paranoia check (non dovrebbe succedere)
        if sample is None:
            raise RuntimeError(f"Dataset ha ritornato None per indice valido {valid_idx}")
        
        return sample
