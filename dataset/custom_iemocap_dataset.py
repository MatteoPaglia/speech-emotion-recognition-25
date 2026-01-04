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
    
    def __init__(self, dataset_root, split='train', transform=None, target_length=3.0, target_sample_rate=16000, target_n_fft=2048, target_hop_length=512, target_n_mels=128, use_silence_trimming=True, use_avg_audio=True):
        """
        Args:
            dataset_root (str): Path to IEMOCAP dataset root folder
            split (str): 'train', 'validation', or 'test'
            transform (callable, optional): Optional transform (non usato, qui per compatibilità)
            target_length (float): Lunghezza target in secondi
            target_sample_rate (int): Sample rate (16000 Hz)
            target_n_fft (int): FFT size (2048)
            target_hop_length (int): Hop length (512)
            target_n_mels (int): Numero di mel bins (128)
            use_silence_trimming (bool): Se applicare silence trimming
            use_avg_audio (bool): Se True usa media della durata, se False usa massimo
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform
        self.use_silence_trimming = use_silence_trimming
        self.use_avg_audio = use_avg_audio
        
        # Audio processing parameters 
        self.target_sample_rate = target_sample_rate
        self.n_fft = target_n_fft
        self.hop_length = target_hop_length
        self.n_mels = target_n_mels
        
        # Default target_samples
        self.target_samples = int(target_length * self.target_sample_rate)  # 48000
        
        # Flag e variabili per la durata POST-TRIMMING (calcolate lazy al primo uso)
        self.trimmed_stats_computed = False
        self.mean_trimmed_samples = None
        self.max_trimmed_samples = None
        
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
        
        
        # Collect all samples (folder_id, sample_id)
        self.samples = self._collect_samples()
        
        # Split into train/test
        self._split_dataset()
     
        
        print(f"✅ Dataset initialized: {len(self.samples)} {split} samples")
    
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
        """
        samples = []
        data_dir = self.dataset_root
        
        print(f"🔍 Raccogliendo campioni audio...")
        
        # Itera su tutte le sessioni
        for folder in sorted(data_dir.iterdir()):
            if folder.is_dir() and folder.name.startswith("Session"):
                folder_id = folder.name[-1]  # Estrai ID sessione (es. '1' da 'Session1')
                
                # Raccogli campioni da folder improvvisati
                sentence_folder = folder / "sentences" / "wav"
                
                if sentence_folder.exists():
                    for folder_sample in sentence_folder.iterdir():
                        if "impro" in folder_sample.name:
                            actor = folder_sample.name.split("_")[0][-1]  # Estrai M o F
                            impro_id = folder_sample.name.split("impro")[1][:2]  # Estrai ID impro
                            
                            # Itera su tutti i file WAV in questa cartella improvvisata
                            for sample_file in sorted(folder_sample.glob("*.wav")):
                                sample_id = sample_file.stem  # es. 'Ses04F_impro06_F002'
                                audio_path = sample_file
                                
                                # Cerca l'etichetta nel dizionario pre-caricato
                                if sample_id in self.label_dict:
                                    sample_data = {
                                        'session_id': folder_id,
                                        'audio_path': audio_path,
                                        'sample_id': sample_id,
                                        'actor': actor,
                                        'impro_id': impro_id,
                                        'label': self.label_dict[sample_id]  # Accesso O(1) al dict
                                    }
                                    samples.append(sample_data)
        
        print(f"✅ Raccolti {len(samples)} campioni audio")
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
        Processa la waveform completa:
        FASE 1: Trim silenzio dalle estremità (calcolo media pst-trimming se necessario)
        FASE 2: Center crop o padding alla durata media POST-TRIMMING
        
        Args:
            waveform (torch.Tensor): Tensore audio [1, num_samples]
        
        Returns:
            torch.Tensor: Waveform processata [1, mean_trimmed_samples]
        """
        # FASE 1: TRIM SILENZIO
        if self.use_silence_trimming:
            try:
                waveform_np = waveform.numpy()[0]
                trimmed, _ = librosa.effects.trim(waveform_np, top_db=40, ref=np.max)
                waveform = torch.from_numpy(trimmed).unsqueeze(0).float()
            except Exception as e:
                print(f"⚠️  Errore nel trim_silence: {e}")
        
        # CALCOLA MEDIA E MASSIMO POST-TRIMMING (una sola volta, lazy)
        if not self.trimmed_stats_computed:
            print(f"\n📊 Calcolando statistiche durata POST-TRIMMING per split '{self.split}'...")
            total_samples = 0
            max_samples = 0
            count = 0
            
            for idx, sample in enumerate(self.samples):
                try:
                    wf, sr = torchaudio.load(str(sample['audio_path']))
                    
                    if sr != self.target_sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
                        wf = resampler(wf)
                    
                    if wf.shape[0] > 1:
                        if self.use_avg_audio:
                            wf = torch.mean(wf, dim=0, keepdim=True)
                        else:
                            wf = torch.max(wf, dim=0, keepdim=True)[0].unsqueeze(0)
                    
                    # TRIM SILENZIO
                    if self.use_silence_trimming:
                        wf_np = wf.numpy()[0]
                        trimmed_wf, _ = librosa.effects.trim(wf_np, top_db=40, ref=np.max)
                        wf = torch.from_numpy(trimmed_wf).unsqueeze(0).float()
                    
                    total_samples += wf.shape[1]
                    max_samples = max(max_samples, wf.shape[1])
                    count += 1
                    
                    if (idx + 1) % max(1, len(self.samples) // 5) == 0:
                        print(f"   {idx + 1}/{len(self.samples)} file processati...")
                except Exception:
                    continue
            
            if count > 0:
                self.mean_trimmed_samples = total_samples // count
                self.max_trimmed_samples = max_samples
                avg_seconds = self.mean_trimmed_samples / self.target_sample_rate
                max_seconds = self.max_trimmed_samples / self.target_sample_rate
                print(f"✅ Media: {avg_seconds:.2f}s ({self.mean_trimmed_samples} campioni)")
                print(f"✅ Massimo: {max_seconds:.2f}s ({self.max_trimmed_samples} campioni)\n")
            else:
                self.mean_trimmed_samples = int(3.0 * self.target_sample_rate)
                self.max_trimmed_samples = int(3.0 * self.target_sample_rate)
                print(f"❌ Calcolo fallito, usando default 3.0s\n")
            
            self.trimmed_stats_computed = True
        
        # FASE 2: CENTER CROP o PADDING basato su media o massimo
        c, n = waveform.shape
        target_len = self.max_trimmed_samples if not self.use_avg_audio else self.mean_trimmed_samples
        
        if n > target_len:
            # Audio troppo lungo: CENTER CROP (prendi la parte centrale)
            start = (n - target_len) // 2
            waveform = waveform[:, start:start+target_len]
                
        elif n < target_len:
            # Audio troppo corto: PADDING
            padding = target_len - n
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
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
        
        try:
            # 1. Load Audio
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            # Resample
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)
                
            # Mono check
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 2. Process Waveform (trim silenzio + center crop/padding)
            waveform = self._process_waveform(waveform)
            
            # 3. Mel Spectrogram
            mel_spec = self.mel_transform(waveform)
            log_mel_spec = self.db_transform(mel_spec)
            
            # 4. Normalization (Z-score)
            mean = log_mel_spec.mean()
            std = log_mel_spec.std()
            log_mel_spec = (log_mel_spec - mean) / (std + 1e-6)
            
            # 5. Return dictionary
            return {
                'mel_spectrogram': log_mel_spec,       # Tensor [1, 128, T]
                'label': emotion_id,                    # Class ID (0-3)
                'emotion': emotion_label,               # Emotion name
                'actor_id': speaker_id                  # Str: 'M' o 'F'
            }
        
        except Exception as e:
            print(f"⚠️  Errore caricamento file {audio_path}: {e}")
            # Ritorna None per segnalare errore - verrà filtrato dal FilteredDatasetWrapper
            return None

