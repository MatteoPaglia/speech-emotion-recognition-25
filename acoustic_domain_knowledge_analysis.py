"""
Speech Emotion Recognition - Acoustic Domain Knowledge Analysis
================================================================

Script modulare per analizzare le caratteristiche acustiche di file audio in SER.
Genera grafici di alta qualità per il domain knowledge analysis.

Autore: ML Vision Project
Data: 2026
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Tuple
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAZIONE PERCORSI E PARAMETRI GLOBALI
# ============================================================================

# Auto-detect dataset paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / "data"

# Percorsi di input - Rilevati automaticamente dai dataset
RAVDESS_DIR = DATA_DIR / "ravdess"
IEMOCAP_DIR = DATA_DIR / "iemocap" / "IEMOCAP_full_release"

# Trova automaticamente i primi file audio disponibili
def find_sample_audio(dataset_dir, dataset_name):
    """Trova il primo file audio disponibile nel dataset."""
    if not dataset_dir.exists():
        print(f"❌ Cartella {dataset_name} non trovata: {dataset_dir}")
        return None
    
    # RAVDESS: cerca in Actor_01
    if "ravdess" in str(dataset_dir).lower():
        actor_dir = dataset_dir / "Actor_01"
        if actor_dir.exists():
            wav_files = list(actor_dir.glob("*.wav"))
            if wav_files:
                return str(wav_files[0])
    
    # IEMOCAP: cerca in Session1
    if "IEMOCAP" in str(dataset_dir):
        session_dir = dataset_dir / "Session1" / "sentences" / "wav"
        if session_dir.exists():
            for impro_dir in session_dir.iterdir():
                if impro_dir.is_dir():
                    wav_files = list(impro_dir.glob("*.wav"))
                    if wav_files:
                        return str(wav_files[0])
    
    return None

# Carica i file
AUDIO_FILE_RAVDESS = find_sample_audio(RAVDESS_DIR, "RAVDESS")
AUDIO_FILE_IEMOCAP = find_sample_audio(IEMOCAP_DIR, "IEMOCAP")

# Percorso di output per le figure
OUTPUT_DIR = PROJECT_ROOT / "acoustic_analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# PARAMETRI STFT E MEL-SPECTROGRAM - ESPLICITI E DICHIARATI
# ============================================================================

# Questi parametri sono CRITICI per la risoluzione tempo-frequenza
SAMPLE_RATE = 16000  # Hz - Allineato al preprocessing del training
N_FFT = 1024  # Dimensione della finestra FFT
HOP_LENGTH = 256  # Numero di campioni tra finestre successive
WINDOW_LENGTH = 1024  # Lunghezza della finestra (uguale a N_FFT per Hann)
N_MELS = 128  # Numero di bande mel

# Parametri per allineamento preprocessing
TARGET_DURATION = 3.0  # secondi
TARGET_SAMPLES = int(TARGET_DURATION * SAMPLE_RATE)  # 48000 @ 16kHz

# Parametri supplementari per feature extraction
FMIN = 0.0  # Hz - coerente con default torchaudio
FMAX = SAMPLE_RATE / 2  # Hz - Nyquist

# Parametri per pitch extraction (Yin algorithm)
FMIN_PITCH = 75  # Hz
FMAX_PITCH = 400  # Hz - limita alla voce umana

# ============================================================================
# FUNZIONE 1: ANALISI STFT E LOG-MEL SPECTROGRAM
# ============================================================================

def print_stft_parameters():
    """
    Stampa in console i parametri STFT espliciti e la loro interpretazione acustica.
    
    Questi parametri controllano il compromesso tra:
    - Risoluzione frequenziale: Δf = SAMPLE_RATE / N_FFT (Hz/bin)
    - Risoluzione temporale: Δt = HOP_LENGTH / SAMPLE_RATE (secondi/frame)
    """
    print("\n" + "="*80)
    print("PARAMETRI STFT E LOG-MEL SPECTROGRAM")
    print("="*80)
    print(f"Sample Rate:           {SAMPLE_RATE} Hz")
    print(f"FFT Size (N_FFT):      {N_FFT}")
    print(f"Hop Length:            {HOP_LENGTH} samples")
    print(f"Window Length:         {WINDOW_LENGTH} samples")
    print(f"Window Type:           Hann (default in librosa)")
    print(f"Mel Bands (N_MELS):    {N_MELS}")
    print(f"Freq Min (fmin):       {FMIN} Hz")
    print(f"Freq Max (fmax):       {FMAX} Hz")
    
    # Calcola e stampa i parametri derivati
    freq_resolution = SAMPLE_RATE / N_FFT
    time_resolution = HOP_LENGTH / SAMPLE_RATE
    nyquist = SAMPLE_RATE / 2
    
    print(f"\n--- PARAMETRI DERIVATI ---")
    print(f"Risoluzione Frequenziale: {freq_resolution:.2f} Hz/bin")
    print(f"Risoluzione Temporale:    {time_resolution*1000:.2f} ms/frame")
    print(f"Frequenza di Nyquist:     {nyquist} Hz")
    print(f"\n--- MOTIVAZIONE SCELTE ---")
    window_ms = (N_FFT / SAMPLE_RATE) * 1000
    hop_ms = (HOP_LENGTH / SAMPLE_RATE) * 1000
    overlap = 1 - (HOP_LENGTH / WINDOW_LENGTH)
    print(f"N_FFT={N_FFT}: Rappresenta ~{window_ms:.1f}ms di segnale (buon compromesso voce)")
    print(f"HOP_LENGTH={HOP_LENGTH}: {hop_ms:.1f}ms tra frame (~{overlap*100:.0f}% overlap)")
    print(f"N_MELS={N_MELS}: Conforme scala Mel biologica umana (udito non lineare)")
    print("="*80 + "\n")


def process_waveform(y: np.ndarray, target_len: int) -> np.ndarray:
    """
    Applica peak-centered crop con shift dinamico + padding finale se necessario.
    """
    n = y.shape[0]
    if n > target_len:
        peak_idx = int(np.argmax(np.abs(y)))
        half_window = target_len // 2
        ideal_start = peak_idx - half_window
        actual_start = max(0, min(ideal_start, n - target_len))
        y = y[actual_start : actual_start + target_len]
    elif n < target_len:
        pad = target_len - n
        y = np.pad(y, (0, pad), mode='constant')
    return y


def load_processed_audio(audio_path: str) -> Tuple[np.ndarray, int]:
    """
    Carica audio, resample a SAMPLE_RATE e applica il crop/padding a 3s.
    """
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    y = process_waveform(y, TARGET_SAMPLES)
    return y, sr


def analyze_logmel_spectrogram(audio_path: str, title: str = "Log-Mel Spectrogram") -> np.ndarray:
    """
    Carica un file audio e genera il Log-Mel Spectrogram.
    
    Parametri:
    -----------
    audio_path : str
        Percorso al file audio
    title : str
        Titolo del plot
        
    Ritorna:
    --------
    S_mel : np.ndarray
        Log-Mel spectrogram normalizzato
    """
    # Carica audio e applica preprocess (allineato al training)
    y, sr = load_processed_audio(audio_path)

    # Applica scala Mel
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WINDOW_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX
    )
    
    # Log scaling (con epsilon per stabilità numerica)
    S_mel = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    img = librosa.display.specshow(S_mel, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                    x_axis='time', y_axis='mel', fmin=FMIN, 
                                    fmax=FMAX, cmap='magma', ax=ax)
    ax.set_title(f'{title}\n(N_MELS={N_MELS}, N_FFT={N_FFT}, HOP_LENGTH={HOP_LENGTH})',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency (Mel)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.set_label('Power (dB)', fontsize=10)
    
    plt.tight_layout()
    
    # Salva figura
    output_file = OUTPUT_DIR / f"01_logmel_spectrogram_{Path(audio_path).stem}.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Salvato: {output_file}")
    plt.close()
    
    return S_mel


# ============================================================================
# FUNZIONE 2: FEATURE EXTRACTION DI BASSO LIVELLO (MULTI-SUBPLOT)
# ============================================================================

def extract_low_level_features(audio_path: str, label: str = "Audio") -> Tuple[np.ndarray, ...]:
    """
    Estrae feature a basso livello allineate al training:
    - Waveform (crop/pad a 3s)
    - Log-Mel spectrogram
    - Mel spectral centroid (Hz)

    Ritorna anche il segnale e sample rate per usi ulteriori.
    """
    y, sr = load_processed_audio(audio_path)

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WINDOW_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)

    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    mel_centroid = np.sum(mel_spec * mel_freqs[:, None], axis=0) / (np.sum(mel_spec, axis=0) + 1e-10)

    return y, sr, log_mel, mel_centroid


def plot_multi_feature_comparison(audio_path_1: str, audio_path_2: str, 
                                   label_1: str = "Domain 1", 
                                   label_2: str = "Domain 2"):
    """
    Genera una figura con multi-subplot mostrando le feature estratte da due file diversi.
    Facilita il confronto visivo tra RAVDESS (pulito) e IEMOCAP (rumoroso/spontaneo).
    
    Parametri:
    -----------
    audio_path_1, audio_path_2 : str
        Percorsi ai due file audio
    label_1, label_2 : str
        Etichette per i due audio (es. "RAVDESS", "IEMOCAP")
    """
    
    # Estrai feature da entrambi i file
    y1, sr1, log_mel1, centroid1 = extract_low_level_features(audio_path_1, label_1)
    y2, sr2, log_mel2, centroid2 = extract_low_level_features(audio_path_2, label_2)
    
    # Crea figura con 4 righe, 2 colonne
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), dpi=150)
    fig.suptitle(f'Log-Mel + Centroid Comparison: {label_1} vs {label_2}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # ========================
    # ROW 1: WAVEFORM ORIGINALE
    # ========================
    time1 = np.linspace(0, len(y1)/sr1, len(y1))
    time2 = np.linspace(0, len(y2)/sr2, len(y2))
    
    axes[0, 0].plot(time1, y1, linewidth=0.7, color='steelblue')
    axes[0, 0].set_title(f'{label_1} - Waveform', fontweight='bold')
    axes[0, 0].set_ylabel('Amplitude', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(time2, y2, linewidth=0.7, color='coral')
    axes[0, 1].set_title(f'{label_2} - Waveform', fontweight='bold')
    axes[0, 1].set_ylabel('Amplitude', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # ========================
    # ROW 2: LOG-MEL SPECTROGRAM
    # ========================
    img1_mel = axes[1, 0].imshow(log_mel1, aspect='auto', origin='lower', cmap='magma')
    axes[1, 0].set_title(f'{label_1} - Log-Mel Spectrogram', fontweight='bold')
    axes[1, 0].set_ylabel('Mel Bin', fontsize=10)
    axes[1, 0].set_xlabel('Time Frame', fontsize=10)
    plt.colorbar(img1_mel, ax=axes[1, 0], label='dB')
    
    img2_mel = axes[1, 1].imshow(log_mel2, aspect='auto', origin='lower', cmap='magma')
    axes[1, 1].set_title(f'{label_2} - Log-Mel Spectrogram', fontweight='bold')
    axes[1, 1].set_ylabel('Mel Bin', fontsize=10)
    axes[1, 1].set_xlabel('Time Frame', fontsize=10)
    plt.colorbar(img2_mel, ax=axes[1, 1], label='dB')
    
    # ========================
    # ROW 3: MEL SPECTRAL CENTROID
    # ========================
    frames1 = np.arange(len(centroid1))
    frames2 = np.arange(len(centroid2))
    times1 = librosa.frames_to_time(frames1, sr=sr1, hop_length=HOP_LENGTH)
    times2 = librosa.frames_to_time(frames2, sr=sr2, hop_length=HOP_LENGTH)
    
    axes[2, 0].plot(times1, centroid1, linewidth=1.5, color='teal', alpha=0.8)
    axes[2, 0].fill_between(times1, centroid1, alpha=0.3, color='teal')
    axes[2, 0].set_title(f'{label_1} - Mel Spectral Centroid', fontweight='bold')
    axes[2, 0].set_ylabel('Frequency (Hz)', fontsize=10)
    axes[2, 0].set_xlabel('Time (s)', fontsize=10)
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(times2, centroid2, linewidth=1.5, color='darkorange', alpha=0.8)
    axes[2, 1].fill_between(times2, centroid2, alpha=0.3, color='darkorange')
    axes[2, 1].set_title(f'{label_2} - Mel Spectral Centroid', fontweight='bold')
    axes[2, 1].set_ylabel('Frequency (Hz)', fontsize=10)
    axes[2, 1].set_xlabel('Time (s)', fontsize=10)
    axes[2, 1].grid(True, alpha=0.3)
    
    # ========================
    # ROW 5: STATISTICHE RIASSUNTIVE
    # ========================
    ax_stats = axes[3, :]
    ax_stats = ax_stats.flatten()
    
    # Nascondi i subplot inutilizzati
    for i in range(len(ax_stats)):
        ax_stats[i].axis('off')
    
    # Crea una tabella riassuntiva
    ax_table = ax_stats[0]
    ax_table.axis('on')
    ax_table.axis('off')
    
    stats_data = [
        ['Metrica', label_1, label_2],
        ['Durata (s)', f'{len(y1)/sr1:.2f}', f'{len(y2)/sr2:.2f}'],
        ['RMS Energy', f'{np.sqrt(np.mean(y1**2)):.4f}', f'{np.sqrt(np.mean(y2**2)):.4f}'],
        ['Mean Mel Centroid (Hz)', f'{np.mean(centroid1):.0f}', f'{np.mean(centroid2):.0f}'],
        ['Std Mel Centroid (Hz)', f'{np.std(centroid1):.0f}', f'{np.std(centroid2):.0f}'],
        ['Mean Log-Mel (dB)', f'{np.mean(log_mel1):.2f}', f'{np.mean(log_mel2):.2f}'],
    ]
    
    table = ax_table.table(cellText=stats_data, cellLoc='center', loc='center',
                          colWidths=[0.35, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Stilizza header
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax_table.text(0.5, 1.15, 'Summary Statistics', ha='center', fontsize=12, 
                  fontweight='bold', transform=ax_table.transAxes)
    
    plt.tight_layout()
    
    # Salva figura
    output_file = OUTPUT_DIR / f"02_feature_comparison_{Path(audio_path_1).stem}_vs_{Path(audio_path_2).stem}.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Salvato: {output_file}")
    plt.close()


# ============================================================================
# FUNZIONE 3: ANALISI CRITICA DEL PEAK-CENTERED CROPPING
# ============================================================================

def extract_f0_and_rms(audio_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estrae F0 (Frequenza Fondamentale) e RMS Energy da un file audio.
    
    Parametri:
    -----------
    audio_path : str
        Percorso al file audio
        
    Ritorna:
    --------
    (f0, rms, times)
    """
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    # Estrai F0 usando il metodo Yin (robusto per voce)
    f0 = librosa.yin(y, fmin=FMIN_PITCH, fmax=FMAX_PITCH, trough_threshold=0.1)
    
    # Estrai RMS Energy
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(S=S)[0]
    
    # Array di tempi
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP_LENGTH)
    
    return f0, rms, times


def plot_peak_centered_analysis(audio_path: str, crop_duration: float = 3.0, 
                                 title: str = "Audio"):
    """
    Analizza il peak-centered cropping:
    - Waveform con picco di ampiezza massima evidenziato
    - F0 (Pitch) sovrapposto
    - RMS Energy sovrapposto
    
    Critica del paper: il picco di volume NON sempre coincide con variazioni prosodiche significative.
    
    Parametri:
    -----------
    audio_path : str
        Percorso al file audio
    crop_duration : float
        Durata del crop in secondi
    title : str
        Titolo del plot
    """
    
    # Carica audio
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    time = np.linspace(0, len(y)/sr, len(y))
    
    # Estrai F0 e RMS
    f0, rms, times_frame = extract_f0_and_rms(audio_path)
    
    # Trova il picco di ampiezza massima (peak volume)
    peak_idx = np.argmax(np.abs(y))
    peak_time = peak_idx / sr
    peak_amplitude = np.abs(y[peak_idx])
    
    # Calcola intervallo di crop
    crop_start = peak_time - crop_duration / 2
    crop_end = peak_time + crop_duration / 2
    crop_start = max(0, crop_start)
    crop_end = min(len(y) / sr, crop_end)
    
    # Crea figura con 2 subplot allineati temporalmente
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=150, sharex=True)
    fig.suptitle(f'{title}\nCritica del Peak-Centered Cropping (durata={crop_duration}s)',
                 fontsize=14, fontweight='bold')
    
    # ========================
    # SUBPLOT 1: WAVEFORM + PEAK INDICATOR
    # ========================
    ax1 = axes[0]
    ax1.plot(time, y, linewidth=0.8, color='steelblue', alpha=0.9, label='Waveform')
    ax1.axvline(peak_time, color='red', linestyle='--', linewidth=2.5, 
                label=f'Peak Volume ({peak_time:.2f}s, Amp={peak_amplitude:.3f})')
    
    # Evidenzia la zona di crop
    ax1.axvspan(crop_start, crop_end, alpha=0.15, color='yellow', label='Crop Region (±1.5s)')
    
    ax1.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    ax1.set_title('Waveform e Localizzazione del Picco di Volume', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ========================
    # SUBPLOT 2: F0 + RMS ENERGY
    # ========================
    ax2 = axes[1]
    
    # Normalizza F0 per visualizzazione (escludendo zeri)
    f0_normalized = np.where(f0 > 0, f0, np.nan)
    ax2_f0 = ax2
    ax2_f0.plot(times_frame, f0_normalized, linewidth=2, color='green', 
               alpha=0.8, label='F0 (Fundamental Frequency)', marker='o', markersize=3)
    ax2_f0.set_ylabel('F0 (Hz)', fontsize=11, fontweight='bold', color='green')
    ax2_f0.tick_params(axis='y', labelcolor='green')
    ax2_f0.grid(True, alpha=0.3)
    
    # RMS su asse secondario
    ax2_rms = ax2.twinx()
    ax2_rms.plot(times_frame, rms, linewidth=2, color='orange', alpha=0.7, 
                label='RMS Energy', linestyle='--', marker='s', markersize=3)
    ax2_rms.set_ylabel('RMS Energy', fontsize=11, fontweight='bold', color='orange')
    ax2_rms.tick_params(axis='y', labelcolor='orange')
    
    # Evidenzia la zona di crop
    ax2.axvspan(crop_start, crop_end, alpha=0.15, color='yellow', label='Crop Region')
    ax2.axvline(peak_time, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
    
    # Legend combinata
    lines1, labels1 = ax2_f0.get_legend_handles_labels()
    lines2, labels2 = ax2_rms.get_legend_handles_labels()
    ax2_f0.legend(lines1 + lines2 + [mpatches.Patch(facecolor='yellow', alpha=0.15, 
                                                      label='Crop Region')],
                 labels1 + labels2 + ['Crop Region'], loc='upper right', fontsize=9)
    
    ax2.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax2.set_title('Contour di Pitch (F0) e Energia RMS - Analisi Prosodica', 
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Salva figura
    output_file = OUTPUT_DIR / f"03_peak_centered_analysis_{Path(audio_path).stem}.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Salvato: {output_file}")
    plt.close()
    
    # Stampa statistiche critiche
    print(f"\n--- ANALISI PEAK-CENTERED CROPPING ---")
    print(f"File: {Path(audio_path).name}")
    print(f"Durata totale: {len(y)/sr:.2f}s")
    print(f"Picco di volume: {peak_time:.2f}s (Ampiezza={peak_amplitude:.4f})")
    print(f"Regione di crop: {crop_start:.2f}s - {crop_end:.2f}s")
    print(f"\nVARIAZIONI PROSODICHE NELLA ZONA DI CROP:")
    print(f"  - F0 minima: {np.nanmin(f0_normalized):.1f} Hz")
    print(f"  - F0 massima: {np.nanmax(f0_normalized):.1f} Hz")
    print(f"  - Variazione F0: {np.nanmax(f0_normalized) - np.nanmin(f0_normalized):.1f} Hz")
    print(f"  - RMS media: {np.mean(rms):.4f}")
    print(f"CRITICA: Se la variazione di F0 è bassa attorno al peak volume,")
    print(f"         il nucleo emotivo potrebbe NON essere catturato dal crop!")
    print("-" * 50 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Funzione principale - esegue tutte le analisi.
    """
    
    print("\n" + "="*80)
    print("SPEECH EMOTION RECOGNITION - ACOUSTIC DOMAIN KNOWLEDGE ANALYSIS")
    print("="*80)
    
    # ========================================================================
    # VERIFICA PERCORSI
    # ========================================================================
    print("\n[VERIFICAZIONE DATASET]")
    print("-" * 80)
    
    print(f"Project Root:    {PROJECT_ROOT}")
    print(f"Data Directory:  {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    print(f"\n✓ RAVDESS Dataset:")
    if RAVDESS_DIR.exists():
        num_files = sum([len(files) for _, _, files in os.walk(RAVDESS_DIR)])
        print(f"  Percorso: {RAVDESS_DIR}")
        print(f"  File: {num_files}")
    else:
        print(f"  ❌ Non trovato!")
    
    print(f"\n✓ IEMOCAP Dataset:")
    if IEMOCAP_DIR.exists():
        num_files = sum([len(files) for _, _, files in os.walk(IEMOCAP_DIR)])
        print(f"  Percorso: {IEMOCAP_DIR}")
        print(f"  File: {num_files}")
    else:
        print(f"  ❌ Non trovato!")
    
    print(f"\n✓ File Audio Selezionati:")
    print(f"  RAVDESS: {AUDIO_FILE_RAVDESS}")
    if AUDIO_FILE_RAVDESS and Path(AUDIO_FILE_RAVDESS).exists():
        print(f"           ✅ Trovato")
    else:
        print(f"           ❌ Non trovato")
    
    print(f"  IEMOCAP: {AUDIO_FILE_IEMOCAP}")
    if AUDIO_FILE_IEMOCAP and Path(AUDIO_FILE_IEMOCAP).exists():
        print(f"           ✅ Trovato")
    else:
        print(f"           ❌ Non trovato")
    
    print("-" * 80)
    
    # ========================================================================
    # STAMPA PARAMETRI STFT
    # ========================================================================
    print_stft_parameters()
    
    # ========================================================================
    # ANALISI LOG-MEL SPECTROGRAMS
    # ========================================================================
    print("[STEP 1] Generazione Log-Mel Spectrograms...")
    print("-" * 80)
    
    if AUDIO_FILE_RAVDESS:
        try:
            analyze_logmel_spectrogram(AUDIO_FILE_RAVDESS, title="RAVDESS - Log-Mel Spectrogram")
        except Exception as e:
            print(f"⚠ Errore nell'analisi RAVDESS: {e}")
    else:
        print(f"⚠ File RAVDESS non disponibile")
    
    if AUDIO_FILE_IEMOCAP:
        try:
            analyze_logmel_spectrogram(AUDIO_FILE_IEMOCAP, title="IEMOCAP - Log-Mel Spectrogram")
        except Exception as e:
            print(f"⚠ Errore nell'analisi IEMOCAP: {e}")
    else:
        print(f"⚠ File IEMOCAP non disponibile")
    
    # ========================================================================
    # FEATURE EXTRACTION E CONFRONTO
    # ========================================================================
    print("\n[STEP 2] Generazione Feature Extraction Multi-Subplot...")
    print("-" * 80)
    
    if AUDIO_FILE_RAVDESS and AUDIO_FILE_IEMOCAP:
        try:
            plot_multi_feature_comparison(AUDIO_FILE_RAVDESS, AUDIO_FILE_IEMOCAP,
                                         label_1="RAVDESS (Pulito)",
                                         label_2="IEMOCAP (Spontaneo/Rumoroso)")
        except Exception as e:
            print(f"⚠ Errore nel confronto feature: {e}")
    else:
        print("⚠ File audio non disponibili per il confronto")
    
    # ========================================================================
    # ANALISI PEAK-CENTERED CROPPING
    # ========================================================================
    print("\n[STEP 3] Analisi del Peak-Centered Cropping...")
    print("-" * 80)
    
    if AUDIO_FILE_RAVDESS:
        try:
            plot_peak_centered_analysis(AUDIO_FILE_RAVDESS, crop_duration=3.0, 
                                       title="RAVDESS - Peak-Centered Cropping Analysis")
        except Exception as e:
            print(f"⚠ Errore nell'analisi RAVDESS: {e}")
    
    if AUDIO_FILE_IEMOCAP:
        try:
            plot_peak_centered_analysis(AUDIO_FILE_IEMOCAP, crop_duration=3.0,
                                       title="IEMOCAP - Peak-Centered Cropping Analysis")
        except Exception as e:
            print(f"⚠ Errore nell'analisi IEMOCAP: {e}")
    
    # ========================================================================
    # RIEPILOGO
    # ========================================================================
    print("\n" + "="*80)
    print(f"✓ ANALISI COMPLETATA")
    print(f"✓ Grafici salvati in: {OUTPUT_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
