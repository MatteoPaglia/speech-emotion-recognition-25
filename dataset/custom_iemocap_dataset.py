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
    
    def __init__(self, dataset_root, split='train', transform=None, target_length=3.0, target_sample_rate=16000, target_n_fft=2048, target_hop_length=512, target_n_mels=128, use_silence_trimming=True):
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
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform
        self.use_silence_trimming = use_silence_trimming
        
        # Audio processing parameters (identici a RAVDESS)
        self.target_sample_rate = target_sample_rate
        self.n_fft = target_n_fft
        self.hop_length = target_hop_length
        self.n_mels = target_n_mels
        
        # Default target_samples
        self.target_samples = int(target_length * self.target_sample_rate)  # 48000
        
        # Flag e variabile per la media durata POST-TRIMMING (calcolata lazy al primo uso)
        self.mean_trimmed_samples_computed = False
        self.mean_trimmed_samples = None
        
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
                    
                    # Collect in sentence folder for improvised samples only
                    sentence_folder = folder / "sentences" / "wav"
                    
                    for folder_sample in sentence_folder.iterdir():
                        if "impro" in folder_sample.name:
                            actor = folder_sample.name.split("_")[0][-1]  # Extract actor M or F
                            impro_id = folder_sample.name.split("impro")[1][:2]  # Extract impro ID es. 06
                            
                            # Load label file once per improvisation folder
                            label_folder = folder / "dialog" / "EmoEvaluation"
                            label_file = label_folder / f"{folder_sample.name}.txt" 
                            
                            # Parse the label file to create a mapping: sample_id -> emotion_label
                            sample_labels = {}
                            try:
                                with open(label_file, 'r') as f:
                                    for line in f:
                                        if line.strip():
                                            parts = line.strip().split('\t')
                                            # Example: [6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]
                                            if len(parts) >= 3:
                                                sample_id = parts[1]
                                                emotion_label = parts[2]
                                                
                                                # FILTRO: Solo le 4 emozioni che ci interessano
                                                if emotion_label in self.EMOTION_DICT:
                                                    sample_labels[sample_id] = emotion_label
                            except FileNotFoundError:
                                print(f"      ⚠ Label file not found: {label_file}")
                                continue
                            
                            # Now iterate through all WAV files in this improvisation folder
                            for sample_file in sorted(folder_sample.glob("*.wav")):
                                sample_id = sample_file.stem  # Extract sample ID without extension (e.g., 'Ses04F_impro06_F002')
                                audio_path = sample_file  # Full path to audio file
                                
                                # Check if this sample has a valid emotion label
                                if sample_id in sample_labels:
                                    sample_data = {
                                        'session_id': folder_id,
                                        'audio_path': audio_path,
                                        'sample_id': sample_id,
                                        'actor': actor,
                                        'impro_id': impro_id,
                                        'label': sample_labels[sample_id]
                                    }
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
                    audio, sr = librosa.load(str(sample['audio_path']), sr=self.target_sample_rate)
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
        
        # CALCOLA MEDIA POST-TRIMMING (una sola volta, lazy)
        if not self.mean_trimmed_samples_computed:
            print(f"\n📊 Calcolando media durata POST-TRIMMING per split '{self.split}'...")
            total_samples = 0
            count = 0
            
            for idx, sample in enumerate(self.samples):
                try:
                    wf, sr = torchaudio.load(str(sample['audio_path']))
                    
                    if sr != self.target_sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
                        wf = resampler(wf)
                    
                    if wf.shape[0] > 1:
                        wf = torch.mean(wf, dim=0, keepdim=True)
                    
                    # TRIM SILENZIO
                    if self.use_silence_trimming:
                        wf_np = wf.numpy()[0]
                        trimmed_wf, _ = librosa.effects.trim(wf_np, top_db=40, ref=np.max)
                        wf = torch.from_numpy(trimmed_wf).unsqueeze(0).float()
                    
                    total_samples += wf.shape[1]
                    count += 1
                    
                    if (idx + 1) % max(1, len(self.samples) // 5) == 0:
                        print(f"   {idx + 1}/{len(self.samples)} file processati...")
                except Exception:
                    continue
            
            if count > 0:
                self.mean_trimmed_samples = total_samples // count
                avg_seconds = self.mean_trimmed_samples / self.target_sample_rate
                print(f"✅ Media calcolata: {avg_seconds:.2f}s ({self.mean_trimmed_samples} campioni)\n")
            else:
                self.mean_trimmed_samples = int(3.0 * self.target_sample_rate)
                print(f"❌ Calcolo fallito, usando default 3.0s\n")
            
            self.mean_trimmed_samples_computed = True
        
        # FASE 2: CENTER CROP o PADDING basato sulla media
        c, n = waveform.shape
        target_len = self.mean_trimmed_samples
        
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
        
        # 5. Return dictionary witi no label logic implemented
        return {
            'audio_features': log_mel_spec,      # Tensor [1, 128, T]
            'emotion_id': emotion_id,            # Int: 0-3
            'emotion': emotion_label if self.split != 'train' else None,  # Emotion o None
            'actor_id': speaker_id               # Str: 'M' o 'F'
        }

