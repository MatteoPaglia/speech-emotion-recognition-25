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
import librosa


class CustomIEMOCAPDataset(Dataset):
    
    def __init__(self, dataset_root, split='train', train_ratio=0.8, seed=42, transform=None):
        
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        
        # Audio processing parameters
        self.sample_rate = 16000  # 16 kHz
        self.n_fft = 2048         # FFT size
        self.hop_length = 512     # Hop length for STFT
        self.use_mfcc = False     # Set to True to use MFCC instead of log spectrogram
        
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
        
        # Use dataset_root directly (should already point to IEMOCAP_full_release or equivalent)
        data_dir = self.dataset_root
        
        print(f"📂 Starting to collect samples from: {data_dir}")
        
        # Check if data_dir is empty
        try:
            items = list(data_dir.iterdir())
            print(f"📂 Found {len(items)} items in directory")
            for item in items[:5]:
                print(f"   - {item.name} {'(dir)' if item.is_dir() else '(file)'}")
            if len(items) > 5:
                print(f"   ... and {len(items) - 5} more items")
        except Exception as e:
            print(f"❌ Error reading directory: {e}")
            return samples
        
        # Iterate through all object folders (Session1, Session2, etc.)
        session_count = 0
        for folder in sorted(data_dir.iterdir()):
            if folder.is_dir():
                print(f"   Checking folder: {folder.name}")
                if(folder.name.startswith("Session")):
                    session_count += 1
                    folder_id = folder.name[-1]  # Extract folder ID (e.g., '1' from 'Session1')
                    print(f"\n📁 Processing {folder.name} (Session ID: {folder_id})")
                    
                    # Collect in sentence forlder for improvised samples only
                    sentence_folder = folder / "sentences" / "wav"
                    print(f"   └─ Looking for audio in: {sentence_folder}")
                    
                    for folder_sample in sentence_folder.iterdir():
                        if "impro" in folder_sample.name:
                            print(f"   └─ Found impro folder: {folder_sample.name}")
                            sample_data = {
                                'session_id': folder_id, #es. '1'
                                'audio_path': None, # es. IEMOCAP_full_release/Session4/sentences/MOCAP_hand/Ses04F_impro06/Ses04F_impro06_F002.wav
                                'sample_id': None, # es. 'Ses04F_impro06_F002'
                                'actor': None, #es. 'F'
                                'impro_id': None, #es. '06'
                                'label': None #es. 'happy'
                            }
                            actor = folder_sample.name.split("_")[0][-1]  # Extract actor S or F
                            impro_id = folder_sample.name.split("impro")[-2]  # Extract impro ID es. 06
                            sample_data['actor'] = actor
                            sample_data['impro_id'] = impro_id
                            
                            for sample_file in sorted(folder_sample.glob("*.wav")):
                                sample_id = sample_file.stem  # Extract sample ID without extension
                                sample_folder = sentence_folder / folder_sample.name / sample_file.name  # Extract the full path of the sample file Es. IEMOCAP_full_release/Session4/sentences/MOCAP_hand/Ses04F_impro06/Ses04F_impro06_F002.wav
                                sample_data['audio_path'] = sample_folder
                                sample_data['sample_id'] = sample_id

                            label_folder = folder / "dialog" / "EmoEvaluation"
                            label_file = label_folder / f"{folder_sample.name}.txt" #es. IEMOCAP_full_release/Session4/dialog/EmoEvaluation/Ses01F_impro06.txt
                            print(f"      Looking for label file: {label_file}")
                            #Open Label file and extract label for the sample Ses04F_impro06_F002, search the line that contains the sample_id, split and after the name there is the label
                            try:
                                with open(label_file, 'r') as f:
                                    for line in f:
                                        if sample_id in line:
                                            parts = line.strip().split('\t')
                                            # Example of line : [6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]
                                            emotion_label = parts[2]  # Extract the emotion label
                                            sample_data['label'] = emotion_label
                                            print(f"      ✓ {sample_id} → Label: {emotion_label}")
                                            break
                            except FileNotFoundError:
                                print(f"      ⚠ Label file not found: {label_file}")
                            
                            samples.append(sample_data)
        
        print(f"✅ Collected {len(samples)} samples in total")
        return samples
    
    def _split_dataset(self,session_train=['1','2','3'], session_val=['4'], session_test=['5']):
        """Split dataset into train and test sets."""
        if len(self.samples) == 0:
            raise ValueError("No samples found in dataset!")
        
        train_samples = [s for s in self.samples if s['session_id'] in session_train]
        val_samples = [s for s in self.samples if s['session_id'] in session_val]
        test_samples = [s for s in self.samples if s['session_id'] in session_test]

        if self.split == 'train':
            self.samples = train_samples
        elif self.split == 'val':
            self.samples = val_samples
        else:
            self.samples = test_samples
    
    
    
    def __len__(self):
        """Return the total number of samples in the selected split."""
        return len(self.samples)
    
    def _load_audio_features(self, audio_path):
        """Load audio and compute log spectrogram or MFCC."""
        try:
            # Load audio at 16kHz
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
            
            if self.use_mfcc:
                # Compute MFCC coefficients
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_fft=self.n_fft, 
                                           hop_length=self.hop_length, n_mfcc=13)
                # Convert to torch tensor (13, time_frames)
                features = torch.FloatTensor(mfcc)
            else:
                # Compute STFT (Short-Time Fourier Transform)
                stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
                
                # Get magnitude spectrogram
                magnitude = np.abs(stft)
                
                # Convert to dB scale (log spectrogram)
                log_spectrogram = librosa.power_to_db(magnitude ** 2, ref=np.max)
                
                # Convert to torch tensor and add channel dimension (1, freq_bins, time_frames)
                features = torch.FloatTensor(log_spectrogram).unsqueeze(0)
            
            return features
        
        except Exception as e:
            print(f"Error loading audio from {audio_path}: {e}")
            raise
    
    
    def __getitem__(self, idx):
        """Retrieve a single sample by index."""
        if idx < 0 or idx >= len(self.samples):
            raise IndexError("Index out of range")
        
        sample_info = self.samples[idx]
        audio_path = sample_info['audio_path']
        label = sample_info['label']
        speaker_id = sample_info['actor']
        session_id = sample_info['session_id']
        impro_id = sample_info['impro_id']
        # Load audio features (e.g., MFCCs, spectrograms)
        audio_features = self._load_audio_features(audio_path)
        return {
            'audio_features': audio_features,
            'label': label,
            'speaker_id': speaker_id,
            'session_id': session_id,
            'impro_id': impro_id
        }

