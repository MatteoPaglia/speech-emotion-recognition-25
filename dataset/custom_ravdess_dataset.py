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
import torch
import torchaudio
import librosa
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset
from utils.get_dataset_statistics import print_dataset_stats


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
    
    def __init__(self, dataset_root, split='train', transform=None, target_length=3.0, target_sample_rate=16000, target_n_fft=2048, target_hop_length=512, target_n_mels=128, use_silence_trimming=True, use_avg_audio=True, spec_freq_mask=12, spec_time_mask=15):
        """
        Args:
            dataset_root (str): Path to the RAVDESS dataset root folder
            split (str): 'train', 'validation', or 'test'
            transform (callable, optional): Optional transform to be applied on audio waveform
            use_silence_trimming (bool): Se True, applica silence trimming ai dati
            spec_freq_mask (int): Parametro per FrequencyMasking in SpecAugment
            spec_time_mask (int): Parametro per TimeMasking in SpecAugment
            
        Split fisso:
            - Train: Actors 01-20 (10M + 10F)
            - Validation: Actors 21-22 (1M + 1F)
            - Test: Actors 23-24 (1M + 1F)
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform
        self.use_silence_trimming = use_silence_trimming
        self.use_avg_audio = use_avg_audio
        self.spec_freq_mask = spec_freq_mask
        self.spec_time_mask = spec_time_mask

        #hyperparameters per l'estrazione delle feature
        self.target_sample_rate = target_sample_rate
        self.n_fft = target_n_fft
        self.hop_length = target_hop_length
        self.n_mels = target_n_mels

        # Default target_samples 
        self.target_samples = int(target_length * self.target_sample_rate) # 48000 (3 secondi @ 16kHz)
        
        # Trasformazione MelSpectrogram (Identica a IEMOCAP per coerenza)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        
        # SpecAugment per Training (maschera parti dello spettrogramma)
        # Solo per training, non per validation/test
        if self.split == 'train':
            # STRATEGIA LEGGERA: Ridotto per preservare feature sottili (es. Sad)
            self.spec_augment = torch.nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=spec_freq_mask), 
                torchaudio.transforms.TimeMasking(time_mask_param=spec_time_mask),    
            )
        else:
            self.spec_augment = None
        
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
    
    def _validate_audio_file(self, audio_path, min_duration=0.5, max_duration=30.0):
        """
        Valida l'integrità di un file audio usando LIBROSA (no FFmpeg richiesto).
        
        Args:
            audio_path (Path): Percorso al file audio
            min_duration (float): Durata minima in secondi
            max_duration (float): Durata massima in secondi
        
        Returns:
            tuple: (is_valid: bool, error_message: str or None)
        """
        try:
            # Verifica che il file esista
            if not audio_path.exists():
                return False, "File non esiste"
            
            # Verifica che sia leggibile
            if not os.access(audio_path, os.R_OK):
                return False, "File non leggibile"
            
            # Carica con librosa (no FFmpeg richiesto!)
            waveform, sample_rate = librosa.load(str(audio_path), sr=None)
            
            # Verifica dimensioni
            if len(waveform) == 0:
                return False, "Waveform vuota"
            
            # Calcola durata
            duration = len(waveform) / sample_rate
            
            # Verifica intervallo durata
            if duration < min_duration:
                return False, f"Troppo corto ({duration:.2f}s < {min_duration}s)"
            if duration > max_duration:
                return False, f"Troppo lungo ({duration:.2f}s > {max_duration}s)"
            
            # Verifica che non sia tutto silenzio
            if np.max(np.abs(waveform)) < 1e-6:
                return False, "Audio tutto silenzio"
            
            return True, None
            
        except Exception as e:
            return False, f"Errore librosa: {str(e)}"
       
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
            
            # 8. VALIDAZIONE: Verifica integrità file audio
            is_valid, error_msg = self._validate_audio_file(audio_file)
            if not is_valid:
                continue
            
            # 9. Se passa tutti i filtri, aggiungi ai samples
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
        - Training: Actors 01-20 (10 maschi dispari + 10 femmine pari)
        - Validation: Actors 21-22 (1 maschio dispari + 1 femmina pari)
        - Test: Actors 23-24 (1 maschio dispari + 1 femmina pari)
        
        Questo garantisce:
        - Speaker-independent (nessun attore ripetuto tra i set)
        - Split deterministico (sempre uguale)
        - Bilanciamento perfetto di genere
        """
        if len(self.samples) == 0:
            raise ValueError("No samples found in dataset!")
        
        # Split fisso basato su ID attori
        train_actors = set(range(1, 21))      # 1-20
        validation_actors = {21, 22}          # 21-22
        test_actors = {23, 24}                # 23-24
        
        print(f"📊 Statistiche del dataset RAVDESS:")
        # Filtra i samples in base agli attori
        if self.split == 'train':
            self.samples = [s for s in self.samples if int(s['metadata']['actor']) in train_actors]
            print_dataset_stats(self.samples, name="RAVDESS TRAINING SET")
            
        elif self.split == 'validation':
            self.samples = [s for s in self.samples if int(s['metadata']['actor']) in validation_actors]
            print_dataset_stats(self.samples, name="RAVDESS VALIDATION SET")
            
        elif self.split == 'test':
            self.samples = [s for s in self.samples if int(s['metadata']['actor']) in test_actors]
            print_dataset_stats(self.samples, name="RAVDESS TEST SET")
            
            
        else:
            raise ValueError(f"Split non valido: {self.split}. Usa 'train', 'validation' o 'test'.")
    
    

    def _process_waveform(self, waveform):
        """
        Processa la waveform per renderla esattamente 3 secondi:
        - Audio troppo lungo: CENTER CROP (prendi parte centrale)
        - Audio troppo corto: ZERO PADDING (aggiungi silenzio)
        
        Args:
            waveform (torch.Tensor): Tensore audio [1, num_samples]
        
        Returns:
            torch.Tensor: Waveform processata [1, target_samples] (48000 campioni @ 16kHz = 3s)
        """
        c, n = waveform.shape
        target_len = self.target_samples  # 48000
        
        if n > target_len:
            # Audio troppo lungo: CENTER CROP (prendi la parte centrale)
            start = (n - target_len) // 2
            waveform = waveform[:, start:start+target_len]
                
        elif n < target_len:
            # Audio troppo corto: ZERO PADDING (aggiungi silenzio alla fine)
            padding_needed = target_len - n
            waveform = torch.nn.functional.pad(waveform, (0, padding_needed), mode='constant', value=0)
            
        return waveform
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        audio_path = sample_info['path']
        metadata = sample_info['metadata']
        
        # 1. Load Audio con librosa (evita dipendenza FFmpeg)
        waveform_np, sample_rate = librosa.load(str(audio_path), sr=None)
        waveform = torch.from_numpy(waveform_np).unsqueeze(0).float()
        
        # Resample
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)
            
        # Mono check
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 2. Process Waveform (trim silenzio + center crop/padding)
        waveform = self._process_waveform(waveform)
        
        # 3. AUGMENTATION WAVEFORM (Solo per Training - Speech Emotion Recognition Safe)
        if self.split == 'train':
            # A. Gaussian Noise Addition (50% probabilità) - CRITICO per Sad/Neutral
            # Serve a rompere la simmetria tra silenzio triste e silenzio neutro
            # Aiuta anche a rendere il modello robusto al rumore ambientale
            if random.random() < 0.5:
                noise_level = random.uniform(0.001, 0.005)  # Livello basso ma variabile
                noise = torch.randn_like(waveform) * noise_level
                waveform = waveform + noise
            
            # B. Amplitude Gain (50% probabilità)
            # Simula variazioni di volume/distanza dal microfono (preserva emozione)
            if random.random() < 0.5:
                gain = random.uniform(0.8, 1.2)
                waveform = waveform * gain
                # Clipping per evitare distorsione
                waveform = torch.clamp(waveform, -1.0, 1.0)
            
            # C. Time Shift / Rolling (50% probabilità) - Aiuta a non memorizzare l'inizio esatto
            # Sposta temporalmente l'audio (simula timing variabile dell'emozione)
            if random.random() < 0.5:
                shift_amt = int(random.random() * self.target_sample_rate * 0.1)  # Max 0.1s
                waveform = torch.roll(waveform, shift_amt, dims=1)
        
        # 4. Mel Spectrogram
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.db_transform(mel_spec)
        
        # 4.5. SpecAugment (Solo per Training)
        # Maschera parti casuali dello spettrogramma per aumentare robustezza
        if self.spec_augment is not None:
            log_mel_spec = self.spec_augment(log_mel_spec)
        
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



    def __len__(self):
        """Return the total number of samples in the selected split."""
        return len(self.samples)