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
import torch
import torchaudio
import librosa
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset
from utils.get_dataset_statistics import print_iemocap_stats


class CustomIEMOCAPDataset(Dataset):
    
    # Mapping delle emozioni IEMOCAP - SOLO le 4 che ci interessano
    EMOTION_DICT = {
        'neu': 'neutral',    # Neutral
        'hap': 'happy',      # Happiness
        'sad': 'sad',        # Sadness
        'ang': 'angry',      # Anger
        'exc': 'happy'       # Excitement became 'happy'
    }
    
    # Mapping per emotion_id (0-indexed, come in RAVDESS)
    EMOTION_ID_MAP = {
        'neu': 0,   # neutral
        'hap': 1,   # happy
        'exc': 1,   # excitement became happy
        'sad': 2,   # sad
        'ang': 3    # angry
    }
    
    def __init__(self, dataset_root, split='train', transform=None, target_length=3.0, target_sample_rate=16000, target_n_fft=2048, target_hop_length=512, target_n_mels=128, spec_freq_mask=12, spec_time_mask=15):
        """
        Args:
            dataset_root (str): Path to IEMOCAP dataset root folder
            split (str): 'train', 'validation', or 'test'
            transform (callable, optional): Optional transform (non usato, qui per compatibilità)
            target_length (float): Lunghezza target in secondi (default: 3.0s)
            target_sample_rate (int): Sample rate (16000 Hz)
            target_n_fft (int): FFT size (2048)
            target_hop_length (int): Hop length (512)
            target_n_mels (int): Numero di mel bins (128)
            spec_freq_mask (int): Parametro per FrequencyMasking in SpecAugment
            spec_time_mask (int): Parametro per TimeMasking in SpecAugment
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform
        self.spec_freq_mask = spec_freq_mask
        self.spec_time_mask = spec_time_mask
        
        # Audio processing parameters 
        self.target_sample_rate = target_sample_rate
        self.n_fft = target_n_fft
        self.hop_length = target_hop_length
        self.n_mels = target_n_mels
        
        # Finestra fissa a 3 secondi (identica a RAVDESS)
        self.target_samples = int(target_length * self.target_sample_rate)  # 48000 @ 16kHz = 3s
        
        # Pre-carica tutte le etichette: {sample_id: emotion_label}
        self.label_dict = self._preload_all_labels()
        print(f"✅ Caricate {len(self.label_dict)} etichette")
        
        # Trasformazione MelSpectrogram (identica a RAVDESS per coerenza)
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
            self.spec_augment = torch.nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=spec_freq_mask), 
                torchaudio.transforms.TimeMasking(time_mask_param=spec_time_mask),    
            )
        else:
            self.spec_augment = None
        
        
        # Collect all samples (folder_id, sample_id)
        self.samples = self._collect_samples()
        
        # Split into train/test
        self._split_dataset()
     
        
        print(f"✅ Dataset initialized: {len(self.samples)} {split} samples")
    
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
    
    def _preload_all_labels(self):
        """
        Pre-carica TUTTE le etichette da tutti i file di valutazione.
        Crea una struttura: {sample_id: emotion_label}
        Eseguito UNA SOLA VOLTA durante l'inizializzazione.
        
        Returns:
            dict: {sample_id (str): emotion_label (str)}
        """
        label_dict = {}
        data_dir = self.dataset_root
        
        # Itera su tutte le sessioni
        for folder in sorted(data_dir.iterdir()):
            if folder.is_dir() and folder.name.startswith("Session"):
                label_folder = folder / "dialog" / "EmoEvaluation"
                
                # Itera su tutti i file .txt di valutazione
                if label_folder.exists():
                    for label_file in label_folder.glob("*.txt"):
                        try:
                            with open(label_file, 'r') as f:
                                for line in f:
                                    if line.strip():
                                        parts = line.strip().split('\t')
                                        # Esempio: [6.2901 - 8.2357]\tSes01F_impro01_F000\tneu\t[2.5000, 2.5000, 2.5000]
                                        if len(parts) >= 3:
                                            sample_id = parts[1]  # es. 'Ses01F_impro01_F000'
                                            emotion_label = parts[2]  # es. 'neu'
                                            
                                            # FILTRO: Solo le 4 emozioni che ci interessano
                                            if emotion_label in self.EMOTION_DICT:
                                                label_dict[sample_id] = emotion_label
                        except Exception as e:
                            print(f"      ⚠ Errore lettura {label_file}: {e}")
        
        return label_dict
    
    def _collect_samples(self):
        """
        Collect all available samples from the dataset.
        NOTA: Le etichette vengono cercate nel dizionario pre-caricato (self.label_dict),
        non lette da file durante questa funzione.
        VALIDAZIONE: Skippa i file audio corrotti o non leggibili.
        """
        samples = []
        data_dir = self.dataset_root
        
        print(f"🔍 Raccogliendo campioni audio...")
        
        corrupted_files = []  # Traccia i file corrotti
        skipped_count = 0
        
        # Itera su tutte le sessioni
        for folder in sorted(data_dir.iterdir()):
            if folder.is_dir() and folder.name.startswith("Session"):
                folder_id = folder.name[-1]  # Estrai ID sessione (es. '1' da 'Session1')
                
                # Raccogli campioni improvvisati
                wav_folder = folder / "sentences" / "wav"
                
                if wav_folder.exists():
                    # Itera su tutti i file WAV direttamente nella cartella wav/
                    # (anche dentro sottocartelle per compatibilità)
                    for sample_file in sorted(wav_folder.glob("**/*.wav")):
                        sample_id = sample_file.stem  # es. 'Ses01F_impro01_F000'
                        
                        # Filtra solo i campioni improvvisati (contengono "impro")
                        if "impro" not in sample_id:
                            continue
                        
                        # Cerca l'etichetta nel dizionario pre-caricato
                        if sample_id in self.label_dict:
                            # ✅ VALIDAZIONE: Controlla integrità del file
                            is_valid, error_msg = self._validate_audio_file(sample_file)
                            if not is_valid:
                                corrupted_files.append({
                                    'sample_id': sample_id,
                                    'reason': error_msg,
                                    'path': str(sample_file)
                                })
                                skipped_count += 1
                                continue  # SKIPPA file corrotto
                            
                            # Estrai actor (M o F) e impro_id dal sample_id
                            # es. da 'Ses01F_impro01_F000' estrai 'F' e '01'
                            parts = sample_id.split("_")
                            actor = parts[0][-1]  # Estrai M o F da 'Ses01F'
                            impro_id = parts[1].replace("impro", "")  # Estrai '01' da 'impro01'
                            
                            sample_data = {
                                'session_id': folder_id,
                                'audio_path': sample_file,
                                'sample_id': sample_id,
                                'actor': actor,
                                'impro_id': impro_id,
                                'label': self.label_dict[sample_id]  # Accesso O(1) al dict
                            }
                            samples.append(sample_data)
        
        print(f"✅ Raccolti {len(samples)} campioni audio validi")
        if skipped_count > 0:
            print(f"⚠️  {skipped_count} file corrotti/invalidi SKIPPATI")
            print(f"\n📋 DETTAGLI FILE CORROTTI:")
            for corrupted in corrupted_files[:10]:  # Mostra primi 10
                print(f"   - {corrupted['sample_id']}: {corrupted['reason']}")
            if len(corrupted_files) > 10:
                print(f"   ... e altri {len(corrupted_files) - 10}")
        print(f"   - Solo campioni improvvisati")
        print(f"   - Emozioni: {list(self.EMOTION_DICT.values())}")
        return samples
    
    def _split_dataset(self, session_train=['1','2','3'], session_validation=['4'], session_test=['5']):
        """Split dataset into train and test sets."""
        if len(self.samples) == 0:
            raise ValueError("No samples found in dataset!")
        
        
        
        # Filtra i samples in base alle sessioni
        train_samples = [s for s in self.samples if s['session_id'] in session_train]
        validation_samples = [s for s in self.samples if s['session_id'] in session_validation]
        test_samples = [s for s in self.samples if s['session_id'] in session_test]


        print(f"📊 Statistiche del dataset IEMOCAP:")
        if self.split == 'train':
            self.samples = train_samples
            print_iemocap_stats(self.samples, name="IEMOCAP TRAINING SET")

        elif self.split == 'validation':
            self.samples = validation_samples
            print_iemocap_stats(self.samples, name="IEMOCAP VALIDATION SET")    
          
        elif self.split == 'test':
            self.samples = test_samples
            print_iemocap_stats(self.samples, name="IEMOCAP TEST SET")
        else:
            raise ValueError("Invalid split name. Use 'train', 'validation', or 'test'.")
            
    
    
    
    def __len__(self):
        """Return the total number of samples in the selected split."""
        return len(self.samples)
    
    def _process_waveform(self, waveform):
        """
        Processa la waveform per renderla esattamente 3 secondi (identica a RAVDESS):
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
        """Retrieve a single sample by index."""
        if idx < 0 or idx >= len(self.samples):
            raise IndexError("Index out of range")
        
        sample_info = self.samples[idx]
        audio_path = sample_info['audio_path']
        label = sample_info['label']
        speaker_id = sample_info['actor']
        
        # Map emotion code to emotion label and ID
        emotion_label = self.EMOTION_DICT.get(label, None)
        emotion_id = self.EMOTION_ID_MAP.get(label, None)
        
        # Skip if label is not valid
        if emotion_label is None or emotion_id is None:
            raise ValueError(f"Invalid emotion label: {label}. Only {list(self.EMOTION_DICT.keys())} are supported.")
        
        # 1. Load Audio con librosa (evita dipendenza FFmpeg)
        waveform_np, sample_rate = librosa.load(str(audio_path), sr=None)
        waveform = torch.from_numpy(waveform_np).unsqueeze(0).float()
        
        # Resample se necessario
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)
            
        # Mono check
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 2. Process Waveform (center crop/padding a 3 secondi)
        waveform = self._process_waveform(waveform)
        
        # 3. AUGMENTATION WAVEFORM (Solo per Training - Speech Emotion Recognition Safe)
        if self.split == 'train':
            # A. Gaussian Noise Addition (50% probabilità)
            if random.random() < 0.5:
                noise_level = random.uniform(0.001, 0.005)
                noise = torch.randn_like(waveform) * noise_level
                waveform = waveform + noise
            
            # B. Amplitude Gain (50% probabilità)
            if random.random() < 0.5:
                gain = random.uniform(0.8, 1.2)
                waveform = waveform * gain
                waveform = torch.clamp(waveform, -1.0, 1.0)
            
            # C. Time Shift / Rolling (50% probabilità)
            if random.random() < 0.5:
                shift_amt = int(random.random() * self.target_sample_rate * 0.1)  # Max 0.1s
                waveform = torch.roll(waveform, shift_amt, dims=1)
        
        # 4. Mel Spectrogram
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.db_transform(mel_spec)
        
        # 4.5. SpecAugment (Solo per Training)
        if self.spec_augment is not None:
            log_mel_spec = self.spec_augment(log_mel_spec)
        
        # 5. Normalization (Z-score)
        mean = log_mel_spec.mean()
        std = log_mel_spec.std()
        log_mel_spec = (log_mel_spec - mean) / (std + 1e-6)
        
        # 6. Return dictionary
        return {
            'audio_features': log_mel_spec,  # Tensor [1, 128, T]
            'emotion_id': emotion_id,         # Int (0-3)
            'emotion': emotion_label,         # Str: 'neutral', 'happy', 'sad', 'angry'
            'actor_id': speaker_id            # Str: 'M' o 'F'
        }

