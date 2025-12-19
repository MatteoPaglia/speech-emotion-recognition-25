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
    
    def __init__(self, dataset_root, split='train', transform=None):
        """
        Args:
            dataset_root (str): Path to the RAVDESS dataset root folder
            split (str): 'train', 'validation', or 'test'
            transform (callable, optional): Optional transform to be applied on audio waveform
            
        Split fisso:
            - Train: Actors 01-20 (10M + 10F)
            - Validation: Actors 21-22 (1M + 1F)
            - Test: Actors 23-24 (1M + 1F)
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform

        self.target_sample_rate = 16000
        
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
        
        # Stampa statistiche
        print(f"📊 Totale campioni: {stats['total_samples']}")
        print(f"📊 Totale attori disponibili: {len(stats['available_actors'])}")
        
        print(f"\n🔀 Split fisso predefinito:")
        print(f"   Train:      Actors 01-20 ({stats['train']['males']}M+{stats['train']['females']}F) → {stats['train']['samples']} campioni ({stats['train']['percentage']:.1f}%) | Lunghezza media: {stats['train']['avg_audio_length']:.2f}s")
        print(f"   Validation: Actors 21-22 ({stats['validation']['males']}M+{stats['validation']['females']}F) → {stats['validation']['samples']} campioni ({stats['validation']['percentage']:.1f}%) | Lunghezza media: {stats['validation']['avg_audio_length']:.2f}s")
        print(f"   Test:       Actors 23-24 ({stats['test']['males']}M+{stats['test']['females']}F) → {stats['test']['samples']} campioni ({stats['test']['percentage']:.1f}%) | Lunghezza media: {stats['test']['avg_audio_length']:.2f}s")
        
        print(f"\n👥 Attori assegnati:")
        print(f"   Train:      {sorted(stats['train']['actors'])}")
        print(f"   Validation: {sorted(stats['validation']['actors'])}")
        print(f"   Test:       {sorted(stats['test']['actors'])}")
        
        # Filtra i samples in base agli attori
        if self.split == 'train':
            self.samples = stats['samples_by_split']['train']
        elif self.split == 'validation':
            self.samples = stats['samples_by_split']['validation']
        elif self.split == 'test':
            self.samples = stats['samples_by_split']['test']
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
        import torchaudio
        
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
                    waveform, sr = torchaudio.load(sample['path'])
                    # Lunghezza in secondi
                    length = waveform.shape[1] / sr
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
    
    
    
    def __len__(self):
        """Return the total number of samples in the selected split."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample by index.
        
        Returns:
            dict with keys:
                - 'audio': Audio waveform tensor [1, num_samples]
                - 'sample_rate': Sample rate of the audio
                - 'emotion': Emotion label (string)
                - 'emotion_id': Emotion ID (int, 0-indexed)
                - 'actor': Actor ID (int)
                - 'path': Path to the audio file
                - 'metadata': Full metadata dict
        """
        # Get sample info
        sample = self.samples[idx]
        audio_path = sample['path']
        metadata = sample['metadata']
        
        # Load audio file using torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate
            )
            waveform = resampler(waveform)
            sample_rate = self.target_sample_rate
        
        # Convert to mono if stereo (average channels)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Apply custom transform if provided
        if self.transform is not None:
            waveform = self.transform(waveform)
        
        # Map emotion code to 0-indexed ID
        # '01' -> 0, '03' -> 1, '04' -> 2, '05' -> 3
        emotion_code = metadata['emotion']
        emotion_mapping = {'01': 0, '03': 1, '04': 2, '05': 3}
        emotion_id = emotion_mapping[emotion_code]
        
        return {
            'audio': waveform,                      # [1, num_samples]
            'sample_rate': sample_rate,             # int
            'emotion': metadata['emotion_label'],    # string: 'neutral', 'happy', 'sad', 'angry'
            'emotion_id': emotion_id,               # int: 0, 1, 2, 3
            'actor': int(metadata['actor']),        # int: 1-24
            'intensity': int(metadata['intensity']), # int: 1 or 2
            'path': str(audio_path),                # string path
            'metadata': metadata                     # full metadata dict
        }