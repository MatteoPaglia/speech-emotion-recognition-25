"""
Custom Dataset for IEMOCAP 

This module implements a PyTorch Dataset class for loading IEMOCAP dataset samples
including only the necessary data for emotion recognition tasks:
- Audio features (e.g., MFCCs, spectrograms)    
- Emotion labels
- Speaker IDs
- Session IDs
It supports train/test splitting, data caching, and efficient data loading.
For this task only improvised audio samples are considered.

Esempio di file : IEMOCAP_full_release/Session4/sentences/MOCAP_hand/Ses04F_impro06/Ses04F_impro06_F002.txt
mi serve tutta la cartella perchè ha tralasciato cartelle importanti ? 
"""

import os
import yaml
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms

#once the requirements are installed in colab it should work
from sklearn.model_selection import train_test_split # type: ignore


class CustomIEMOCAPDataset(Dataset):
    
    def __init__(self, dataset_root, split='train', train_ratio=0.8, seed=42, transform=None):
        
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        
        #fare un file di config per gli hyperparameters?

        # Define audio transformations
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
        
        
        # Collect all samples (folder_id, sample_id)
        self.samples = self._collect_samples()
        
        # Split into train/test
        self._split_dataset()
     
        
        print(f"✅ Dataset initialized: {len(self.samples)} {split} samples")
    
    def _collect_samples(self):
        """Collect all available samples from the dataset."""
        samples = []
        data_dir =  "directory of dataset"
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Iterate through all object folders (01, 02, etc.)
        #for folder in sorted(data_dir.iterdir()):
            #if folder.is_dir():
                
                
                
        
        return samples
    
    def _split_dataset(self):
        """Split dataset into train and test sets."""
        if len(self.samples) == 0:
            raise ValueError("No samples found in dataset!")
        
        # Stratified split by folder_id to ensure all objects in both splits
        folder_ids = [s[0] for s in self.samples]
        
        train_samples, test_samples = train_test_split(
            self.samples,
            train_size=self.train_ratio,
            random_state=self.seed,
            stratify=folder_ids
        )
        
        if self.split == 'train':
            self.samples = train_samples
        else:
            self.samples = test_samples
    
    
    
    def __len__(self):
        """Return the total number of samples in the selected split."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Retrieve a single sample by index."""
       
        return None

