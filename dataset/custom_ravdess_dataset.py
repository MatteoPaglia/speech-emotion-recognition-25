"""
Custom Dataset for RAVDESS 

This module implements a PyTorch Dataset class for loading RAVDESS dataset samples
including only the necessary data for emotion recognition tasks:
- Audio features (e.g., MFCCs, spectrograms)    
- Emotion labels
- Speaker IDs
- Session IDs
It supports train/test splitting, data caching, and efficient data loading.
For this task only acted audio samples are considered.

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


class CustomRAVDESSDataset(Dataset):

    # Mapping SOLO delle 4 emozioni che ci interessano
    EMOTION_DICT = {
        '01': 'neutral',   # Neutral
        '03': 'happy',     # Happiness
        '04': 'sad',       # Sadness
        '05': 'angry'      # Anger
    }
    
    # Filtri per RAVDESS
    MODALITY_AUDIO_ONLY = '03'  # Solo audio (no video)
    VOCAL_CHANNEL_SPEECH = '01'  # Solo speech (no song)
    
    def __init__(self, dataset_root, split='train', validation_ratio=0.1, test_ratio=0.1, seed=42, transform=None):
        """
        Args:
            dataset_root (str): Path to the RAVDESS dataset root folder
            split (str): 'train', 'validation', or 'test'
            validation_ratio (float): Proportion of actors for validation (default: 0.1)
            test_ratio (float): Proportion of actors for test (default: 0.1)
            seed (int): Random seed for reproducibility
            transform (callable, optional): Optional transform to be applied on audio
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
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



    def _parse_filename(self, filename):
        """
        Parse RAVDESS filename structure:
        Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
        
        Example: 03-01-06-01-02-01-12.wav
        Returns dict con metadata o None se non valido
        """
        # Estrai solo il nome senza estensione
        name = filename.stem if isinstance(filename, Path) else Path(filename).stem
        
        # Split per '-'
        parts = name.split('-')
        
        # Deve avere esattamente 7 parti
        if len(parts) != 7:
            return None
        
        return {
            'modality': parts[0],           # 03 = audio-only
            'vocal_channel': parts[1],      # 01 = speech, 02 = song
            'emotion': parts[2],            # 01-08 emotion codes
            'intensity': parts[3],          # 01 = normal, 02 = strong
            'statement': parts[4],          # 01 o 02
            'repetition': parts[5],         # 01 o 02
            'actor': parts[6]               # 01-24
        }
    
    
    def _collect_samples(self):
        """
        Collect all available samples from the dataset.
        Filters:
        - Audio-only (modality 03)
        - Speech only (vocal_channel 01)
        - Only emotions: Neutral, Happy, Sad, Angry
        """
        samples = []
    
        # 1. Verifica che la directory del dataset esista
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Data directory not found: {self.dataset_root}")
        
        # 2. Cerca tutti i file .wav nel dataset (ricorsivamente)
        # RAVDESS può avere struttura: ravdess/Actor_XX/*.wav
        audio_files = list(self.dataset_root.rglob('*.wav'))
        
        print(f"🔍 Trovati {len(audio_files)} file audio totali nel dataset")
        
        # 3. Itera su ogni file audio trovato
        for audio_file in audio_files:
            
            # 4. Parsing del filename per estrarre metadata
            metadata = self._parse_filename(audio_file)
            
            # Se il parsing fallisce, salta questo file
            if metadata is None:
                continue
            
            # 5. FILTRO 1: Solo audio-only (modality = 03)
            if metadata['modality'] != self.MODALITY_AUDIO_ONLY:
                continue
            
            # 6. FILTRO 2: Solo speech (vocal_channel = 01, no song)
            if metadata['vocal_channel'] != self.VOCAL_CHANNEL_SPEECH:
                continue
            
            # 7. FILTRO 3: Solo le 4 emozioni che ci interessano
            if metadata['emotion'] not in self.EMOTION_DICT:
                continue
            
            # 8. Se passa tutti i filtri, aggiungi ai samples
            # Aggiungi anche l'etichetta testuale dell'emozione
            metadata['emotion_label'] = self.EMOTION_DICT[metadata['emotion']]
            
            samples.append({
                'path': audio_file,
                'metadata': metadata
            })
        
        print(f"✅ Dopo i filtri: {len(samples)} samples validi")
        print(f"   - Audio-only (modality 03)")
        print(f"   - Speech-only (vocal_channel 01)")
        print(f"   - Emotions: {list(self.EMOTION_DICT.values())}")
        
        return samples
    

    def _split_dataset(self):
        """
        Split dataset into train/validation/test sets usando speaker-independent split.
        Caratteristiche:
        - Nessun attore ripetuto tra i set (speaker-independent)
        - Circa 10% campioni in validation e test (basato su validation_ratio e test_ratio)
        - Bilanciamento per genere (maschi: attori dispari, femmine: attori pari)
        """
        if len(self.samples) == 0:
            raise ValueError("No samples found in dataset!")
        
        # Raggruppa campioni per attore
        from collections import defaultdict
        actor_samples = defaultdict(list)
        for sample in self.samples:
            actor_id = int(sample['metadata']['actor'])
            actor_samples[actor_id].append(sample)
        
        # Conta campioni per attore
        actor_counts = {actor: len(samples) for actor, samples in actor_samples.items()}
        total_samples = len(self.samples)
        
        # Separa attori per genere (dispari=maschi, pari=femmine)
        male_actors = [a for a in actor_counts.keys() if a % 2 == 1]  # Odd = male
        female_actors = [a for a in actor_counts.keys() if a % 2 == 0]  # Even = female
        
        print(f"📊 Totale campioni: {total_samples}")
        print(f"📊 Totale attori: {len(actor_samples)}")
        print(f"   👨 Maschi (dispari): {sorted(male_actors)}")
        print(f"   👩 Femmine (pari): {sorted(female_actors)}")
        
        # Funzione helper per splittare un gruppo di attori
        def split_actors_by_samples(actors_list, target_ratio_test, target_ratio_val):
            """Split actors cercando di raggiungere i target ratio sui campioni."""
            # Shuffle con seed
            np.random.seed(self.seed)
            shuffled = actors_list.copy()
            np.random.shuffle(shuffled)
            
            # Calcola target campioni per questo gruppo
            total_group_samples = sum(actor_counts[a] for a in actors_list)
            target_test = int(total_group_samples * target_ratio_test)
            target_val = int(total_group_samples * target_ratio_val)
            
            test_set = []
            val_set = []
            train_set = []
            
            test_count = 0
            val_count = 0
            
            for actor in shuffled:
                n_samples = actor_counts[actor]
                
                if test_count < target_test:
                    test_set.append(actor)
                    test_count += n_samples
                elif val_count < target_val:
                    val_set.append(actor)
                    val_count += n_samples
                else:
                    train_set.append(actor)
            
            return train_set, val_set, test_set
        
        # Split separato per maschi e femmine (per bilanciamento genere)
        train_male, val_male, test_male = split_actors_by_samples(
            male_actors, self.test_ratio, self.validation_ratio
        )
        train_female, val_female, test_female = split_actors_by_samples(
            female_actors, self.test_ratio, self.validation_ratio
        )
        
        # Combina i set
        train_actors = set(train_male + train_female)
        validation_actors = set(val_male + val_female)
        test_actors = set(test_male + test_female)
        
        # Conta campioni effettivi per ogni set
        train_samples_list = [s for s in self.samples if int(s['metadata']['actor']) in train_actors]
        val_samples_list = [s for s in self.samples if int(s['metadata']['actor']) in validation_actors]
        test_samples_list = [s for s in self.samples if int(s['metadata']['actor']) in test_actors]
        
        print(f"\n🔀 Split risultante:")
        print(f"   Train: {len(train_actors)} attori ({len(train_male)}M+{len(train_female)}F), {len(train_samples_list)} campioni ({len(train_samples_list)/total_samples*100:.1f}%)")
        print(f"   Validation: {len(validation_actors)} attori ({len(val_male)}M+{len(val_female)}F), {len(val_samples_list)} campioni ({len(val_samples_list)/total_samples*100:.1f}%)")
        print(f"   Test: {len(test_actors)} attori ({len(test_male)}M+{len(test_female)}F), {len(test_samples_list)} campioni ({len(test_samples_list)/total_samples*100:.1f}%)")
        
        print(f"\n👥 Attori assegnati:")
        print(f"   Train: {sorted(train_actors)}")
        print(f"   Validation: {sorted(validation_actors)}")
        print(f"   Test: {sorted(test_actors)}")
        
        # Filtra i samples in base agli attori
        if self.split == 'train':
            self.samples = train_samples_list
        elif self.split == 'validation':
            self.samples = val_samples_list
        elif self.split == 'test':
            self.samples = test_samples_list
        else:
            raise ValueError(f"Split non valido: {self.split}. Usa 'train', 'validation' o 'test'.")
    
    
    
    def __len__(self):
        """Return the total number of samples in the selected split."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Retrieve a single sample by index."""
        return None

