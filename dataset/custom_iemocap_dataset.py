import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path

class CustomIEMOCAPDataset(Dataset):
    
    # [cite_start]Mapping Emozioni (Include il merging 'exc' -> 'happy' [cite: 43])
    EMOTION_DICT = {
        'neu': 'neutral',
        'hap': 'happy',
        'exc': 'happy',  # MERGE RICHIESTO DAL DOCUMENTO
        'sad': 'sad',
        'ang': 'angry'
    }
    
    EMOTION_ID_MAP = {
        'neutral': 0,
        'happy': 1,
        'sad': 2,
        'angry': 3
    }

    def __init__(self, dataset_root, split='train', target_length=3.0):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.sample_rate = 16000
        self.target_samples = int(target_length * self.sample_rate) # 48000 samples [cite: 63]

        # Configurazione MelSpectrogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        # 1. Colleziona TUTTI i dati
        self.samples = self._collect_samples()
        
        # 2. Applica lo split (Train/Val/Test)
        self._split_dataset() 
        
        # Check di sicurezza post-split
        if len(self.samples) == 0:
            raise ValueError(f"Nessun sample trovato per lo split '{split}'. Controlla il percorso {self.dataset_root}")
            
        # Nessuna stampa di statistiche qui (per pulizia)

    def _collect_samples(self):
        samples = []
        
        # Verifica esistenza root
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Cartella non trovata: {self.dataset_root}")

        # Itera sulle sessioni (Session1, Session2...)
        # Usa glob per trovare le cartelle SessionX
        session_dirs = sorted(list(self.dataset_root.glob("Session*")))
        
        if not session_dirs:
            # Fallback: prova a cercare dentro una sottocartella comune se l'utente ha sbagliato path
            session_dirs = sorted(list(self.dataset_root.glob("*/Session*")))

        for session_dir in session_dirs:
            if not session_dir.is_dir(): continue
            
            session_id = session_dir.name[-1] # '1' da 'Session1'
            wav_root = session_dir / "sentences" / "wav"
            label_root = session_dir / "dialog" / "EmoEvaluation"
            
            if not wav_root.exists() or not label_root.exists():
                continue

            # Itera sulle sottocartelle delle frasi (es. Ses01F_impro01)
            for folder_sample in wav_root.iterdir():
                # [cite_start]FILTRO 1: Solo sessioni improvvisate ("impro") [cite: 27]
                if "impro" not in folder_sample.name:
                    continue
                
                # Parsing info base dal nome cartella
                parts = folder_sample.name.split('_') # ['Ses01F', 'impro01']
                actor_code = parts[0][-1] # 'F'
                
                # File delle label corrispondente
                label_file = label_root / f"{folder_sample.name}.txt"
                if not label_file.exists(): continue
                
                # Mappatura Label per questa cartella
                label_map = {}
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            if line.startswith("["): 
                                # Format: [Start - End] ID Emotion ...
                                seg = line.strip().split('\t')
                                if len(seg) >= 3:
                                    wav_name = seg[1]
                                    emotion = seg[2]
                                    label_map[wav_name] = emotion
                except Exception:
                    continue

                # Itera sui file .wav
                for wav_file in sorted(folder_sample.glob("*.wav")):
                    wav_name = wav_file.stem
                    raw_label = label_map.get(wav_name)
                    
                    # FILTRO 2: Solo emozioni valide (neu, hap, exc, sad, ang)
                    if raw_label not in self.EMOTION_DICT:
                        continue
                        
                    samples.append({
                        'audio_path': wav_file,
                        'session_id': session_id, # '1', '2', etc.
                        'actor_gender': actor_code, 
                        'label_raw': raw_label,
                        'sample_id': wav_name
                    })
                    
        return samples

    def _split_dataset(self):
        """
        [cite_start]Split Dataset IEMOCAP per Session ID[cite: 28].
        [cite_start]Train: 1-3 [cite: 30][cite_start], Val: 4 [cite: 31][cite_start], Test: 5[cite: 32].
        """
        train_sessions = {'1', '2', '3'}
        val_sessions = {'4'}
        test_sessions = {'5'}
        
        if self.split == 'train':
            target_sessions = train_sessions
        elif self.split == 'validation':
            target_sessions = val_sessions
        elif self.split == 'test':
            target_sessions = test_sessions
        else:
            raise ValueError(f"Split non valido: {self.split}")
            
        # Filtra la lista self.samples in-place
        self.samples = [s for s in self.samples if s['session_id'] in target_sessions]

    def _process_waveform(self, waveform):
        """Gestione lunghezza fissa 3s [cite: 63]"""
        c, n = waveform.shape
        target = self.target_samples
        
        if n > target:
            if self.split == 'train':
                # Random crop per training
                start = torch.randint(0, n - target, (1,)).item()
            else:
                # Center crop per val/test
                start = (n - target) // 2
            waveform = waveform[:, start:start+target]
        elif n < target:
            # Padding con zeri
            padding = target - n
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 1. Load Audio
        try:
            waveform, sr = torchaudio.load(str(item['audio_path']))
        except Exception as e:
            # Fallback di sicurezza per file corrotti (opzionale)
            print(f"Error loading {item['audio_path']}: {e}")
            waveform = torch.zeros(1, self.target_samples)
            sr = self.sample_rate

        # Resample se necessario
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Converti a mono se stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 2. Process (Fixed Length 3s)
        waveform = self._process_waveform(waveform)

        # 3. MelSpectrogram
        spec = self.mel_transform(waveform)
        spec = self.db_transform(spec)
        
        # 4. Normalize (Z-score)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        
        # Labels
        label_str = self.EMOTION_DICT[item['label_raw']] # Mappa 'exc' -> 'happy'
        label_id = self.EMOTION_ID_MAP[label_str]
        
        # Costruzione ID attore composito (es. '1F') per analisi future
        actor_id_composite = item['session_id'] + item['actor_gender']

        return {
            'audio_features': spec,
            'emotion_id': label_id,
            'emotion': label_str,
            'actor_id': actor_id_composite 
        }

    def __len__(self):
        return len(self.samples)