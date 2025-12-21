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
import torchaudio
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
        '02': 'neutral',   # Calm -> Viene mappato a Neutral
        '03': 'happy',     # Happiness
        '04': 'sad',       # Sadness
        '05': 'angry'      # Anger
    }

    EMOTION_ID_MAP = {
        'neutral': 0,
        'happy': 1,
        'sad': 2,
        'angry': 3
    }
    
    # Filtri per RAVDESS
    MODALITY_AUDIO_ONLY = '03'  # Solo audio (no video)
    VOCAL_CHANNEL_SPEECH = '01'  # Solo speech (no song)
    
    def __init__(self, dataset_root, split='train', target_length=3.0):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.target_sample_rate = 16000
        self.target_samples = int(target_length * self.target_sample_rate) # 48000
        
        # Trasformazione MelSpectrogram (Identica a IEMOCAP per coerenza)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        
        self.samples = self._collect_samples()
        self._split_dataset()



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
        
        return samples
    

    def _split_dataset(self):
        """
        Split dataset into train/validation/test sets con split fisso basato su ID attori.
        
        Split predefinito:
        - Training: Actors 01-20 (10 maschi dispari + 10 femmine pari) [cite: 22]
        - Validation: Actors 21-22 (1 maschio dispari + 1 femmina pari) [cite: 23]
        - Test: Actors 23-24 (1 maschio dispari + 1 femmina pari) [cite: 24]
        """
        if len(self.samples) == 0:
            raise ValueError("No samples found in dataset!")
        
        # Definizione Split Attori
        train_actors = set(range(1, 21))
        val_actors = {21, 22}
        test_actors = {23, 24}
        
        # Selezione del target in base allo split richiesto
        if self.split == 'train':
            target_actors = train_actors
        elif self.split == 'validation':
            target_actors = val_actors
        elif self.split == 'test':
            target_actors = test_actors
        else:
            raise ValueError(f"Split non valido: {self.split}")
            
        # Filtraggio: Mantieni solo i sample degli attori nel target set
        self.samples = [
            s for s in self.samples 
            if int(s['metadata']['actor']) in target_actors
        ]
    
    
    
    def __len__(self):
        """Return the total number of samples in the selected split."""
        return len(self.samples)
    
    def _process_waveform(self, waveform):
        # Target fisso a 3 secondi (48000 samples a 16kHz)
        target_len = 48000 
        c, n = waveform.shape

        if n > target_len:
            # CASO AUDIO LUNGO: Taglio intelligente
            if self.split == 'train':
                # Random Crop: In training non perdo nulla su più epoche
                max_start = n - target_len
                start = torch.randint(0, max_start, (1,)).item()
                waveform = waveform[:, start:start+target_len]
            else:
                # Center Crop: In test prendo la parte centrale (più rappresentativa)
                start = (n - target_len) // 2
                waveform = waveform[:, start:start+target_len]
                
        elif n < target_len:
            # CASO AUDIO CORTO: Padding (necessario ma minimizzato)
            padding = target_len - n
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        return waveform
    
    

    def __getitem__(self, idx):
            sample_info = self.samples[idx]
            audio_path = sample_info['path']
            metadata = sample_info['metadata']
            
            # 1. Load Audio
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            # Resample
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)
                
            # Mono check
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 2. Fixed Length (3s) - CRUCIALE
            waveform = self._process_waveform(waveform)

            # 3. Augmentation Waveform (Opzionale per Phase 2)
            # if self.split == 'train' ...

            # 4. Mel Spectrogram
            mel_spec = self.mel_transform(waveform)
            log_mel_spec = self.db_transform(mel_spec)
            
            # 5. Normalization (Z-score)
            mean = log_mel_spec.mean()
            std = log_mel_spec.std()
            log_mel_spec = (log_mel_spec - mean) / (std + 1e-6)

            # Labels
            label_str = metadata['emotion_label']     # 'neutral', 'happy'...
            label_id = self.EMOTION_ID_MAP[label_str] # 0, 1, 2, 3

            return {
                'audio_features': log_mel_spec, # Tensor [1, 128, T]
                'emotion_id': label_id,         # Int
                'emotion': label_str,           # Str (utile per debug)
                'actor_id': metadata['actor']
            }
