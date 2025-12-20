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
    
    # Mapping delle emozioni IEMOCAP - SOLO le 4 che ci interessano
    EMOTION_DICT = {
        'neu': 'neutral',    # Neutral
        'hap': 'happy',      # Happiness
        'sad': 'sad',        # Sadness
        'ang': 'angry'       # Anger
    }
    
    # Mapping per emotion_id (0-indexed, come in RAVDESS)
    EMOTION_ID_MAP = {
        'neu': 0,   # neutral
        'hap': 1,   # happy
        'sad': 2,   # sad
        'ang': 3    # angry
    }
    
    def __init__(self, dataset_root, split='train', transform=None):
        
        self.dataset_root = Path(dataset_root)
        self.split = split
        
        # Audio processing parameters
        self.sample_rate = 16000  # 16 kHz
        self.n_fft = 2048         # FFT size
        self.hop_length = 512     # Hop length for STFT
        self.use_mfcc = False     # Set to True to use MFCC instead of log spectrogram

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
        
        print(f"Starting to collect samples from: {data_dir}")
        
        # Iterate through all object folders (Session1, Session2, etc.)
        session_count = 0
        for folder in sorted(data_dir.iterdir()):
            if folder.is_dir():
                if(folder.name.startswith("Session")):
                    session_count += 1
                    folder_id = folder.name[-1]  # Extract folder ID (e.g., '1' from 'Session1')
                    
                    # Collect in sentence forlder for improvised samples only
                    sentence_folder = folder / "sentences" / "wav"
                    
                    for folder_sample in sentence_folder.iterdir():
                        if "impro" in folder_sample.name:
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
                            #Open Label file and extract label for the sample Ses04F_impro06_F002, search the line that contains the sample_id, split and after the name there is the label
                            try:
                                with open(label_file, 'r') as f:
                                    for line in f:
                                        if sample_id in line:
                                            parts = line.strip().split('\t')
                                            # Example of line : [6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]
                                            emotion_label = parts[2]  # Extract the emotion label
                                            
                                            # FILTRO: Solo le 4 emozioni che ci interessano
                                            if emotion_label not in self.EMOTION_DICT:
                                                break
                                            
                                            sample_data['label'] = emotion_label
                                            break
                            except FileNotFoundError:
                                print(f"      ⚠ Label file not found: {label_file}")
                            
                            samples.append(sample_data)
        
        print(f"✅ Collected {len(samples)} samples in total")
        print(f"   - Improvised samples only")
        print(f"   - Emotions: {list(self.EMOTION_DICT.values())}")
        return samples
    
    def _split_dataset(self, session_train=['1','2','3'], session_validation=['4'], session_test=['5']):
        """Split dataset into train and test sets."""
        if len(self.samples) == 0:
            raise ValueError("No samples found in dataset!")
        
        # Calcola statistiche del dataset
        stats = self.dataset_stat(session_train, session_validation, session_test)
        
        # Filtra i samples in base alle sessioni
        train_samples = stats['samples_by_split']['train']
        validation_samples = stats['samples_by_split']['validation']
        test_samples = stats['samples_by_split']['test']

        print(f"📊 Statistiche del dataset IEMOCAP:")
        if self.split == 'train':
            self.samples = train_samples
            print(f"   Train:      Sessions {stats['train']['sessions']} ({stats['train']['speakers']} speakers) → {stats['train']['samples']} campioni ({stats['train']['percentage']:.1f}%) | Lunghezza media: {stats['train']['avg_audio_length']:.2f}s")
            print(f"\n👤 Speaker distribution:")
            print(f"   Train:      M={stats['train']['males']}, F={stats['train']['females']}")

        elif self.split == 'validation':
            self.samples = validation_samples
            print(f"   Validation: Sessions {stats['validation']['sessions']} ({stats['validation']['speakers']} speakers) → {stats['validation']['samples']} campioni ({stats['validation']['percentage']:.1f}%) | Lunghezza media: {stats['validation']['avg_audio_length']:.2f}s")
            print(f"\n👤 Speaker distribution:")
            print(f"   Validation: M={stats['validation']['males']}, F={stats['validation']['females']}")

        else:
            self.samples = test_samples
            print(f"   Test:       Sessions {stats['test']['sessions']} ({stats['test']['speakers']} speakers) → {stats['test']['samples']} campioni ({stats['test']['percentage']:.1f}%) | Lunghezza media: {stats['test']['avg_audio_length']:.2f}s")
            print(f"\n👤 Speaker distribution:")
            print(f"   Test:       M={stats['test']['males']}, F={stats['test']['females']}")
    
    
    def dataset_stat(self, session_train, session_validation, session_test):
        """
        Calcola tutte le statistiche del dataset per i 3 range di divisione per sessione.
        
        Args:
            session_train (list): Lista degli ID di sessione per training
            session_validation (list): Lista degli ID di sessione per validation
            session_test (list): Lista degli ID di sessione per test
        
        Returns:
            dict: Dizionario con tutte le statistiche calcolate
        """
        import librosa
        
        # Sessioni disponibili nei dati
        available_sessions = set([s['session_id'] for s in self.samples])
        
        # Filtra i samples per ogni split
        train_samples_list = [s for s in self.samples if s['session_id'] in session_train]
        validation_samples_list = [s for s in self.samples if s['session_id'] in session_validation]
        test_samples_list = [s for s in self.samples if s['session_id'] in session_test]
        
        # Calcola speaker count e genere per ogni split
        def get_speaker_stats(samples_list):
            """Calcola statistiche degli speaker (M/F)."""
            speakers = set([s['actor'] for s in samples_list])
            males = [s for s in speakers if s == 'M']
            females = [s for s in speakers if s == 'F']
            return len(males), len(females)
        
        def get_audio_length_stats(samples_list):
            """Calcola la lunghezza media dei file audio."""
            if not samples_list:
                return 0.0
            
            total_length = 0.0
            count = 0
            for sample in samples_list:
                try:
                    audio, sr = librosa.load(str(sample['audio_path']), sr=self.sample_rate)
                    # Lunghezza in secondi
                    length = len(audio) / sr
                    total_length += length
                    count += 1
                except:
                    continue
            
            return total_length / count if count > 0 else 0.0
        
        train_m, train_f = get_speaker_stats(train_samples_list)
        validation_m, validation_f = get_speaker_stats(validation_samples_list)
        test_m, test_f = get_speaker_stats(test_samples_list)
        
        # Calcola lunghezze medie audio
        train_avg_length = get_audio_length_stats(train_samples_list)
        validation_avg_length = get_audio_length_stats(validation_samples_list)
        test_avg_length = get_audio_length_stats(test_samples_list)
        
        total_samples = len(self.samples)
        
        return {
            'total_samples': total_samples,
            'available_sessions': available_sessions,
            'train': {
                'sessions': session_train,
                'speakers': train_m + train_f,
                'males': train_m,
                'females': train_f,
                'samples': len(train_samples_list),
                'percentage': len(train_samples_list) / total_samples * 100 if total_samples > 0 else 0,
                'avg_audio_length': train_avg_length
            },
            'validation': {
                'sessions': session_validation,
                'speakers': validation_m + validation_f,
                'males': validation_m,
                'females': validation_f,
                'samples': len(validation_samples_list),
                'percentage': len(validation_samples_list) / total_samples * 100 if total_samples > 0 else 0,
                'avg_audio_length': validation_avg_length
            },
            'test': {
                'sessions': session_test,
                'speakers': test_m + test_f,
                'males': test_m,
                'females': test_f,
                'samples': len(test_samples_list),
                'percentage': len(test_samples_list) / total_samples * 100 if total_samples > 0 else 0,
                'avg_audio_length': test_avg_length
            },
            'samples_by_split': {
                'train': train_samples_list,
                'validation': validation_samples_list,
                'test': test_samples_list
            }
        }
    
    
    
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
        
        # Map emotion code to emotion label and ID
        emotion_label = self.EMOTION_DICT.get(label, None)
        emotion_id = self.EMOTION_ID_MAP.get(label, None)
        
        # Skip if label is not valid
        if emotion_label is None or emotion_id is None:
            raise ValueError(f"Invalid emotion label: {label}. Only {list(self.EMOTION_DICT.keys())} are supported.")
        
        return {
            'audio_features': audio_features,
            'emotion': emotion_label,           # string: 'neutral', 'happy', 'sad', 'angry'
            'emotion_id': emotion_id,           # int: 0-3 (same as RAVDESS)
            'speaker_id': speaker_id,           # string: 'M' o 'F'
            'session_id': session_id,           # string: '1'-'5'
            'impro_id': impro_id,               # string: es. '06'
            'label': label                      # original label code
        }

