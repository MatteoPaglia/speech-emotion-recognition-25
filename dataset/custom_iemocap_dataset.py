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