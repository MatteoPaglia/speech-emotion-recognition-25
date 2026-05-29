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
    
    def __init__(self, dataset_root, allowed_speakers=None, is_train=True, transform=None, target_length=3.0, target_sample_rate=16000, target_n_fft=1024, target_hop_length=256, target_n_mels=128, spec_freq_mask=12, spec_time_mask=15, pseudo_labels_dict=None, add_gaussian_noise_snr=None):
        """
        Args:
            dataset_root (str): Path to IEMOCAP dataset root folder
            allowed_speakers (list or set): List of speaker IDs to include (e.g., ['1F', '1M']). Returns all if None.
            is_train (bool): If True, applies data augmentation
            transform (callable, optional): Optional transform (non usato, qui per compatibilità)
            target_length (float): Lunghezza target in secondi (default: 3.0s)
            target_sample_rate (int): Sample rate (16000 Hz)
            target_n_fft (int): FFT size (1024)
            target_hop_length (int): Hop length (256)
            target_n_mels (int): Numero di mel bins (128)
            spec_freq_mask (int): Parametro per FrequencyMasking in SpecAugment
            spec_time_mask (int): Parametro per TimeMasking in SpecAugment
            pseudo_labels_dict (dict): Maps sample_id to integer pseudo label id for pseudo supervision
            add_gaussian_noise_snr (tuple): E.g. (10, 20) for SNR range of additive gaussian noise.
        """
        self.dataset_root = Path(dataset_root)
        self.allowed_speakers = set(allowed_speakers) if allowed_speakers is not None else None
        self.is_train = is_train
        self.transform = transform
        self.spec_freq_mask = spec_freq_mask
        self.spec_time_mask = spec_time_mask
        self.pseudo_labels_dict = pseudo_labels_dict
        self.add_gaussian_noise_snr = add_gaussian_noise_snr
        
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
            win_length=1024,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        
        # SpecAugment per Training (maschera parti dello spettrogramma)
        # Solo per training, non per validation/test
        if self.is_train:
            self.spec_augment = torch.nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=spec_freq_mask), 
                torchaudio.transforms.TimeMasking(time_mask_param=spec_time_mask),    
            )
        else:
            self.spec_augment = None
        
        
        # Collect all samples (folder_id, sample_id)
        self.samples = self._collect_samples()
        
        print(f"📊 Statistiche del dataset IEMOCAP:")
        dataset_name = "IEMOCAP TRAINING SET" if self.is_train else "IEMOCAP EVALUATION SET"
        print_iemocap_stats(self.samples, name=dataset_name)
     
        
        print(f"✅ Dataset initialized: {len(self.samples)} samples")
    
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
                            
                            speaker_id = folder_id + actor  # es. '1F', '2M'

                            # 8.5 FILTRO SPEAKER
                            if self.allowed_speakers is not None and speaker_id not in self.allowed_speakers:
                                continue

                            sample_data = {
                                'session_id': folder_id,
                                'audio_path': sample_file,
                                'sample_id': sample_id,
                                'actor': actor,
                                'speaker_id': speaker_id,
                                'impro_id': impro_id,
                                'label': self.label_dict[sample_id]  # Accesso O(1) al dict
                            }
                            
                            # Se fornito il dizionario pseudo-labels e il sample ha una label valida
                            if self.pseudo_labels_dict is not None:
                                if sample_id in self.pseudo_labels_dict:
                                    sample_data['pseudo_label'] = self.pseudo_labels_dict[sample_id]
                                else:
                                    continue # Skip sample if it wasn't confidently predicted by teacher
                                    
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
    
    @staticmethod
    def get_all_speakers(dataset_root):
        """
        Scansione rapida della cartella dataset_root per restituire la lista di 
        tutti gli speaker disponibili e i file audio per i campioni IEMOCAP improvvisati.
        
        Returns:
            audio_files (list of str): Percorsi ai file audio.
            speaker_ids (list of str): ID dello speaker (es '1F', '5M') associato ad ogni file.
        """
        import os
        from pathlib import Path
        
        dataset_root = Path(dataset_root)
        audio_files = []
        speaker_ids = []
        
        if not dataset_root.exists():
            return audio_files, speaker_ids
            
        for folder in sorted(dataset_root.iterdir()):
            if folder.is_dir() and folder.name.startswith("Session"):
                folder_id = folder.name[-1] # Session ID (1-5)
                wav_folder = folder / "sentences" / "wav"
                
                if wav_folder.exists():
                    for sample_file in sorted(wav_folder.glob("**/*.wav")):
                        sample_id = sample_file.stem
                        if "impro" in sample_id:
                            parts = sample_id.split("_")
                            actor = parts[0][-1]
                            speaker_id = folder_id + actor
                            
                            audio_files.append(str(sample_file))
                            speaker_ids.append(speaker_id)
                            
        return audio_files, speaker_ids

    def __len__(self):
        """Return the total number of samples in the selected split."""
        return len(self.samples)
    
    def _process_waveform(self, waveform):
        """
        Processa la waveform usando Peak-Centered Crop con Shift Dinamico:
        - Trova il picco massimo di volume nell'audio.
        - Cerca di prendere i 3 secondi (target_len) centrati sul picco.
        - Se il picco è troppo vicino a un bordo, fa slittare la finestra verso 
          il lato opposto per catturare audio reale e azzerare il padding inutile.
        - Applica padding solo se l'audio totale è < 3 secondi.
        """
        c, n = waveform.shape
        target_len = self.target_samples  # 48000 (3 secondi)
        
        if n > target_len:
            # 1. Trova l'indice del picco massimo (guardando il valore assoluto dell'ampiezza)
            peak_idx = torch.argmax(torch.abs(waveform[0])).item()
            
            # 2. Calcola lo start_idx "ideale" per tenere il picco esattamente al centro
            half_window = target_len // 2
            ideal_start = peak_idx - half_window
            
            # 3. APPLICA LO SHIFT DINAMICO (La regola che hai richiesto)
            # - Se ideal_start < 0 (picco a inizio file), blocca la partenza a 0.
            # - Se ideal_start + target_len > n (picco a fine file), arretra la 
            #   partenza al punto esatto (n - target_len) per includere tutto l'audio finale.
            actual_start = max(0, min(ideal_start, n - target_len))
            
            # 4. Ritaglia la porzione
            waveform = waveform[:, actual_start : actual_start + target_len]
            
        elif n < target_len:
            # Se l'audio intero dura meno di 3 secondi, il padding è purtroppo inevitabile.
            # Aggiungiamo silenzio alla fine.
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
        if self.is_train:
            # A. Additive Gaussian Noise with SNR (Se configurato per Noisy Student)
            if self.add_gaussian_noise_snr is not None:
                # noise in dB
                snr = random.uniform(self.add_gaussian_noise_snr[0], self.add_gaussian_noise_snr[1])
                signal_power = torch.mean(waveform ** 2)
                noise_power = signal_power / (10 ** (snr / 10.0))
                noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
                waveform = waveform + noise
            # A. Gaussian Noise Addition Classico (50% probabilità)
            elif random.random() < 0.5:
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
        ret_dict = {
            'sample_id': sample_info['sample_id'],
            'audio_features': log_mel_spec,  # Tensor [1, 128, T]
            'emotion_id': emotion_id,         # Int (0-3)
            'emotion': emotion_label,         # Str: 'neutral', 'happy', 'sad', 'angry'
            'actor_id': speaker_id            # Str: 'M' o 'F'
        }
        
        if 'pseudo_label' in sample_info:
            ret_dict['pseudo_emotion_id'] = sample_info['pseudo_label']  # Passed already as integer/id
            
        return ret_dict

