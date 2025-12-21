import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path

class CustomIEMOCAPDataset(Dataset):
    
    # Mapping aggiornato secondo 
    EMOTION_DICT = {
        'neu': 'neutral',
        'hap': 'happy',
        'exc': 'happy',  # <--- MERGE RICHIESTO DAL DOCUMENTO
        'sad': 'sad',
        'ang': 'angry'
    }
    
    EMOTION_ID_MAP = {
        'neu': 0,
        'hap': 1,
        'exc': 1,        # <--- Anche qui mappiamo a 1 (Happiness)
        'sad': 2,
        'ang': 3
    }

    def __init__(self, dataset_root, split='train', target_length=3.0):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.sample_rate = 16000
        self.target_samples = int(target_length * self.sample_rate) # 48000 samples (3s)

        # Configurazione MelSpectrogram come da PDF 
        # Spostiamo le trasformazioni qui per efficienza e correttezza
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128  # Valore standard, puoi variarlo
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        self.samples = self._collect_samples()
        self._split_dataset()
        
        print(f"✅ Dataset initialized: {len(self.samples)} {split} samples")

    # ... [Il tuo metodo _collect_samples originale era corretto, mantienilo] ...
    # Ricorda solo di aggiornare la logica di filtro per includere anche 'exc'
    # nella parte: if emotion_label not in self.EMOTION_DICT: ...

    def _process_waveform(self, waveform):
        """Taglia o fa padding per avere esattamente 3 secondi (Fixed-length windows )"""
        c, n = waveform.shape
        if n > self.target_samples:
            # Taglio (prendo il centro o l'inizio)
            start = 0
            waveform = waveform[:, start:start+self.target_samples]
        elif n < self.target_samples:
            # Padding (ripeto o aggiungo zeri)
            padding = self.target_samples - n
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        audio_path = sample_info['audio_path']
        label_code = sample_info['label'] # es. 'exc' o 'hap'

        # 1. Caricamento Waveform con Torchaudio (più veloce per PyTorch)
        waveform, sr = torchaudio.load(str(audio_path))
        
        # Resampling se necessario (IEMOCAP è solitamente 16k, ma per sicurezza)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # 2. Lunghezza Fissa (Cruciale per i Batch)
        waveform = self._process_waveform(waveform)

        # 3. Augmentation Waveform (Qui andrebbe il rumore per la Fase 2)
        # if self.split == 'train' and self.augment: ...

        # 4. Calcolo Log-Mel Spectrogram
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.db_transform(mel_spec)

        # 5. Normalizzazione (Z-score richiesta da [cite: 64])
        # Normalizzazione semplice per-sample
        mean = log_mel_spec.mean()
        std = log_mel_spec.std()
        log_mel_spec = (log_mel_spec - mean) / (std + 1e-6)

        # Recupero ID numerico corretto (gestisce 'exc' -> 1)
        label_id = self.EMOTION_ID_MAP[label_code]

        return {
            'audio_features': log_mel_spec, # Tensore [1, 128, T]
            'emotion_id': label_id,         # Tensore scalare o int
            'label_code': label_code,       # Per debug ('exc', 'hap', etc.)
            'sample_id': sample_info['sample_id']
        }