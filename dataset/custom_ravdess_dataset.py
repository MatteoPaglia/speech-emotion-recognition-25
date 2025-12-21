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
from pathlib import Path
from torch.utils.data import Dataset


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
    
    def __init__(self, dataset_root, split='train', transform=None, target_length=3.0, target_sample_rate=16000, target_n_fft=2048, target_hop_length=512, target_n_mels=128, use_silence_trimming=True):
        """
        Args:
            dataset_root (str): Path to the RAVDESS dataset root folder
            split (str): 'train', 'validation', or 'test'
            transform (callable, optional): Optional transform to be applied on audio waveform
            use_silence_trimming (bool): Se True, applica silence trimming ai dati
            
        Split fisso:
            - Train: Actors 01-20 (10M + 10F)
            - Validation: Actors 21-22 (1M + 1F)
            - Test: Actors 23-24 (1M + 1F)
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform
        self.use_silence_trimming = use_silence_trimming

        #hyperparameters per l'estrazione delle feature
        self.target_sample_rate = target_sample_rate
        self.n_fft = target_n_fft
        self.hop_length = target_hop_length
        self.n_mels = target_n_mels

        # Default target_samples 
        self.target_samples = int(target_length * self.target_sample_rate) # 48000
        
        # Flag e variabile per la media durata POST-TRIMMING (calcolata lazy al primo uso)
        self.mean_trimmed_samples_computed = False
        self.mean_trimmed_samples = None
        
        # Trasformazione MelSpectrogram (Identica a IEMOCAP per coerenza)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
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
        validation_actors = set([21, 22])     # 21-22
        test_actors = set([23, 24])           # 23-24
        
        # Calcola statistiche del dataset
        stats = self.dataset_stat(train_actors, validation_actors, test_actors)
        
        print(f"📊 Statistiche del dataset RAVDESS:")
        # Filtra i samples in base agli attori
        if self.split == 'train':
            self.samples = stats['samples_by_split']['train']
            print(f"   Train:      Actors 01-20 ({stats['train']['males']}M+{stats['train']['females']}F) → {stats['train']['samples']} campioni ({stats['train']['percentage']:.1f}%) | Lunghezza media: {stats['train']['avg_audio_length']:.2f}s")
            print(f"\n👥 Attori assegnati:")
            print(f"   Train:      {sorted(stats['train']['actors'])}")
        elif self.split == 'validation':
            self.samples = stats['samples_by_split']['validation']
            print(f"   Validation: Actors 21-22 ({stats['validation']['males']}M+{stats['validation']['females']}F) → {stats['validation']['samples']} campioni ({stats['validation']['percentage']:.1f}%) | Lunghezza media: {stats['validation']['avg_audio_length']:.2f}s")
            print(f"\n👥 Attori assegnati:")
            print(f"   Validation: {sorted(stats['validation']['actors'])}")
        elif self.split == 'test':
            self.samples = stats['samples_by_split']['test']
            print(f"   Test:       Actors 23-24 ({stats['test']['males']}M+{stats['test']['females']}F) → {stats['test']['samples']} campioni ({stats['test']['percentage']:.1f}%) | Lunghezza media: {stats['test']['avg_audio_length']:.2f}s")
            print(f"\n👥 Attori assegnati:")
            print(f"   Test:       {sorted(stats['test']['actors'])}")
            
        else:
            raise ValueError(f"Split non valido: {self.split}. Usa 'train', 'validation' o 'test'.")
    
    def dataset_stat(self, train_actors, validation_actors, test_actors):
        """
        Calcola tutte le statistiche del dataset per i 3 range di divisione.
        
        Args:
            train_actors (set): Set degli ID degli attori di training
            validation_actors (set): Set degli ID degli attori di validation
            test_actors (set): Set degli ID degli attori di test
        
        Returns:
            dict: Dizionario con tutte le statistiche calcolate
        """
        def get_gender_stats(actors_set):
            """Calcola statistiche di genere per un set di attori."""
            males = [a for a in actors_set if a % 2 == 1]
            females = [a for a in actors_set if a % 2 == 0]
            return len(males), len(females), males, females
        
        def get_audio_length_stats(samples_list):
            """Calcola la lunghezza media dei file audio."""
            if not samples_list:
                return 0.0
            
            total_length = 0.0
            count = 0
            errors = 0
            for sample in samples_list:
                try:
                    audio, sr = librosa.load(str(sample['path']), sr=None)
                    # Lunghezza in secondi
                    length = len(audio) / sr
                    total_length += length
                    count += 1
                except Exception as e:
                    errors += 1
                    print(f"⚠️  Errore nel caricamento di {sample['path']}: {e}")
                    continue
            
            if errors > 0:
                print(f"⚠️  {errors}/{len(samples_list)} file non caricati correttamente")
            
            return total_length / count if count > 0 else 0.0
        
        # Attori disponibili nei dati
        available_actors = set([int(s['metadata']['actor']) for s in self.samples])
        
        # Filtra gli attori effettivamente presenti nei dati
        train_actors = train_actors & available_actors
        validation_actors = validation_actors & available_actors
        test_actors = test_actors & available_actors
        
        # Calcola statistiche di genere per ogni set
        train_m, train_f, train_males, train_females = get_gender_stats(train_actors)
        val_m, val_f, val_males, val_females = get_gender_stats(validation_actors)
        test_m, test_f, test_males, test_females = get_gender_stats(test_actors)
        
        # Filtra campioni per ogni set
        train_samples_list = [s for s in self.samples if int(s['metadata']['actor']) in train_actors]
        val_samples_list = [s for s in self.samples if int(s['metadata']['actor']) in validation_actors]
        test_samples_list = [s for s in self.samples if int(s['metadata']['actor']) in test_actors]
        
        # Calcola lunghezze medie audio
        train_avg_length = get_audio_length_stats(train_samples_list)
        val_avg_length = get_audio_length_stats(val_samples_list)
        test_avg_length = get_audio_length_stats(test_samples_list)
        
        total_samples = len(self.samples)
        
        return {
            'total_samples': total_samples,
            'available_actors': available_actors,
            'train': {
                'actors': train_actors,
                'males': train_m,
                'females': train_f,
                'samples': len(train_samples_list),
                'percentage': len(train_samples_list) / total_samples * 100 if total_samples > 0 else 0,
                'avg_audio_length': train_avg_length
            },
            'validation': {
                'actors': validation_actors,
                'males': val_m,
                'females': val_f,
                'samples': len(val_samples_list),
                'percentage': len(val_samples_list) / total_samples * 100 if total_samples > 0 else 0,
                'avg_audio_length': val_avg_length
            },
            'test': {
                'actors': test_actors,
                'males': test_m,
                'females': test_f,
                'samples': len(test_samples_list),
                'percentage': len(test_samples_list) / total_samples * 100 if total_samples > 0 else 0,
                'avg_audio_length': test_avg_length
            },
            'samples_by_split': {
                'train': train_samples_list,
                'validation': val_samples_list,
                'test': test_samples_list
            }
        }
    
    
    

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
                    wf, sr = torchaudio.load(str(sample['path']))
                    
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
        
        # 2. Process Waveform (trim silenzio + center crop/padding)
        waveform = self._process_waveform(waveform)
        
        # 3. Augmentation Waveform (Opzionale per Phase 2)
        # if self.split == 'train' ...
        
        # 5. Mel Spectrogram
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



    def __len__(self):
        """Return the total number of samples in the selected split."""
        return len(self.samples)