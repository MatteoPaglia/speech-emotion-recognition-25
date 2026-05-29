"""
Speech Emotion Recognition - Acoustic Domain Knowledge Analysis
================================================================

Modular script to analyze acoustic features of audio files in SER.
Generates high-quality plots for domain knowledge analysis and sim-to-real gap evaluation.
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
# GLOBAL CONFIGURATION AND PATHS
# ============================================================================

# Auto-detect dataset paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / "data"

# Input paths - Automatically detected from datasets
RAVDESS_DIR = DATA_DIR / "ravdess"
IEMOCAP_DIR = DATA_DIR / "iemocap" / "IEMOCAP_full_release"

def find_sample_audio(dataset_dir, dataset_name):
    """Finds the first available audio file in the dataset directory."""
    if not dataset_dir.exists():
        return None
    
    # RAVDESS: search in Actor_01
    if "ravdess" in str(dataset_dir).lower():
        actor_dir = dataset_dir / "Actor_01"
        if actor_dir.exists():
            wav_files = list(actor_dir.glob("*.wav"))
            if wav_files:
                return str(wav_files[0])
    
    # IEMOCAP: search in Session1
    if "IEMOCAP" in str(dataset_dir):
        session_dir = dataset_dir / "Session1" / "sentences" / "wav"
        if session_dir.exists():
            for impro_dir in session_dir.iterdir():
                if impro_dir.is_dir():
                    wav_files = list(impro_dir.glob("*.wav"))
                    if wav_files:
                        return str(wav_files[0])
    
    return None

# Load files
AUDIO_FILE_RAVDESS = find_sample_audio(RAVDESS_DIR, "RAVDESS")
AUDIO_FILE_IEMOCAP = find_sample_audio(IEMOCAP_DIR, "IEMOCAP")

# Output path for generated figures
OUTPUT_DIR = PROJECT_ROOT / "acoustic_analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# STFT AND MEL-SPECTROGRAM PARAMETERS - EXPLICITLY DECLARED
# ============================================================================

# These parameters critically control the time-frequency resolution trade-off
SAMPLE_RATE = 16000  # Hz - Aligned with training preprocessing
N_FFT = 1024         # FFT window size
HOP_LENGTH = 256     # Number of samples between successive frames
WINDOW_LENGTH = 1024 # Window length (equal to N_FFT for Hann window)
N_MELS = 128         # Number of Mel bands

# Parameters for preprocessing alignment
TARGET_DURATION = 3.0  # seconds
TARGET_SAMPLES = int(TARGET_DURATION * SAMPLE_RATE)  # 48000 @ 16kHz

# Supplementary parameters for feature extraction
FMIN = 0.0               # Hz - consistent with torchaudio default
FMAX = SAMPLE_RATE / 2   # Hz - Nyquist frequency

# Parameters for pitch extraction (Yin algorithm)
FMIN_PITCH = 75   # Hz
FMAX_PITCH = 400  # Hz - limited to human voice range

# ============================================================================
# FUNCTION 1: STFT PARAMETERS LOGGING
# ============================================================================

def print_stft_parameters():
    """
    Prints explicit STFT parameters and their acoustic interpretation to the console.
    """
    print("\n" + "="*80)
    print("STFT AND LOG-MEL SPECTROGRAM PARAMETERS")
    print("="*80)
    print(f"Sample Rate:           {SAMPLE_RATE} Hz")
    print(f"FFT Size (N_FFT):      {N_FFT}")
    print(f"Hop Length:            {HOP_LENGTH} samples")
    print(f"Window Length:         {WINDOW_LENGTH} samples")
    print(f"Window Type:           Hann (librosa default)")
    print(f"Mel Bands (N_MELS):    {N_MELS}")
    print(f"Freq Min (fmin):       {FMIN} Hz")
    print(f"Freq Max (fmax):       {FMAX} Hz")
    
    freq_resolution = SAMPLE_RATE / N_FFT
    time_resolution = HOP_LENGTH / SAMPLE_RATE
    nyquist = SAMPLE_RATE / 2
    
    print(f"\n--- DERIVED PARAMETERS ---")
    print(f"Frequency Resolution: {freq_resolution:.2f} Hz/bin")
    print(f"Time Resolution:      {time_resolution*1000:.2f} ms/frame")
    print(f"Nyquist Frequency:    {nyquist} Hz")
    
    print(f"\n--- ACOUSTIC RATIONALE ---")
    window_ms = (N_FFT / SAMPLE_RATE) * 1000
    hop_ms = (HOP_LENGTH / SAMPLE_RATE) * 1000
    overlap = 1 - (HOP_LENGTH / WINDOW_LENGTH)
    print(f"N_FFT={N_FFT}: Represents ~{window_ms:.1f}ms of signal (optimal for voice harmonics)")
    print(f"HOP_LENGTH={HOP_LENGTH}: {hop_ms:.1f}ms step (~{overlap*100:.0f}% frame overlap)")
    print(f"N_MELS={N_MELS}: Maps to human auditory non-linear perception")
    print("="*80 + "\n")

# ============================================================================
# UTILITY PROCESSING FUNCTIONS
# ============================================================================

def process_waveform(y: np.ndarray, target_len: int) -> np.ndarray:
    """Applies dynamic peak-centered cropping + padding if necessary."""
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
    """Loads audio, resamples to SAMPLE_RATE, and applies 3s crop/padding."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    y = process_waveform(y, TARGET_SAMPLES)
    return y, sr

def extract_low_level_features(audio_path: str) -> Tuple[np.ndarray, ...]:
    """Extracts training-aligned low-level features."""
    y, sr = load_processed_audio(audio_path)

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WINDOW_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)

    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    mel_centroid = np.sum(mel_spec * mel_freqs[:, None], axis=0) / (np.sum(mel_spec, axis=0) + 1e-10)
    
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]

    return y, sr, log_mel, mel_centroid, zcr

# ============================================================================
# FUNCTION 2: MULTI-FEATURE COMPARISON (CROSS-DOMAIN)
# ============================================================================

def plot_multi_feature_comparison(audio_path_1: str, audio_path_2: str, 
                                   label_1: str = "Domain 1", 
                                   label_2: str = "Domain 2"):
    """
    Generates a multi-subplot figure showing extracted features from two different files.
    Facilitates visual comparison between RAVDESS (clean) and IEMOCAP (noisy/spontaneous).
    """
    y1, sr1, log_mel1, centroid1, zcr1 = extract_low_level_features(audio_path_1)
    y2, sr2, log_mel2, centroid2, zcr2 = extract_low_level_features(audio_path_2)
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), dpi=150)
    fig.suptitle(f'Feature Comparison: {label_1} vs {label_2}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # ROW 1: WAVEFORM
    time1 = np.linspace(0, len(y1)/sr1, len(y1))
    time2 = np.linspace(0, len(y2)/sr2, len(y2))
    
    axes[0, 0].plot(time1, y1, linewidth=0.7, color='steelblue')
    axes[0, 0].axhline(0, color='black', linestyle=':', linewidth=1.2, alpha=0.8)
    axes[0, 0].set_title(f'{label_1} - Waveform', fontweight='bold')
    axes[0, 0].set_ylabel('Amplitude', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(time2, y2, linewidth=0.7, color='coral')
    axes[0, 1].axhline(0, color='black', linestyle=':', linewidth=1.2, alpha=0.8)
    axes[0, 1].set_title(f'{label_2} - Waveform', fontweight='bold')
    axes[0, 1].set_ylabel('Amplitude', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # ROW 2: LOG-MEL SPECTROGRAM
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
    
    # ROW 3: MEL SPECTRAL CENTROID
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
    
    # ROW 4: ZERO CROSSING RATE
    axes[3, 0].plot(times1, zcr1, linewidth=1.5, color='purple', alpha=0.8)
    axes[3, 0].fill_between(times1, zcr1, alpha=0.3, color='purple')
    axes[3, 0].set_title(f'{label_1} - Zero Crossing Rate', fontweight='bold')
    axes[3, 0].set_ylabel('Rate', fontsize=10)
    axes[3, 0].set_xlabel('Time (s)', fontsize=10)
    axes[3, 0].grid(True, alpha=0.3)
    
    axes[3, 1].plot(times2, zcr2, linewidth=1.5, color='crimson', alpha=0.8)
    axes[3, 1].fill_between(times2, zcr2, alpha=0.3, color='crimson')
    axes[3, 1].set_title(f'{label_2} - Zero Crossing Rate', fontweight='bold')
    axes[3, 1].set_ylabel('Rate', fontsize=10)
    axes[3, 1].set_xlabel('Time (s)', fontsize=10)
    axes[3, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / f"01_feature_comparison_{Path(audio_path_1).stem}_vs_{Path(audio_path_2).stem}.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[*] Saved Feature Comparison Plot: {output_file}")
    plt.close()

# ============================================================================
# FUNCTION 3: DYNAMIC CROPPING ANALYSIS (PEAK VS PROSODY)
# ============================================================================

def extract_f0_and_rms(audio_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    f0 = librosa.yin(y, fmin=FMIN_PITCH, fmax=FMAX_PITCH, trough_threshold=0.1,
                     frame_length=N_FFT, hop_length=HOP_LENGTH)
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(S=S, frame_length=N_FFT)[0] 
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP_LENGTH)
    return f0, rms, times

def plot_peak_centered_analysis(audio_path: str, dataset_name: str = "Audio"):
    """
    Analyzes the peak-centered cropping heuristic by comparing the absolute amplitude 
    peak location with underlying prosodic contours (F0 and RMS Energy).
    """
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    time = np.linspace(0, len(y)/sr, len(y))
    
    f0, rms, times_frame = extract_f0_and_rms(audio_path)
    
    peak_idx = np.argmax(np.abs(y))
    peak_time = peak_idx / sr
    peak_amplitude = np.abs(y[peak_idx])
    
    crop_duration = 3.0
    crop_start = max(0, peak_time - crop_duration / 2)
    crop_end = min(len(y) / sr, peak_time + crop_duration / 2)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=150, sharex=True)
    fig.suptitle(f'{dataset_name} - Peak-Centered Cropping Critique', fontsize=14, fontweight='bold')
    
    # SUBPLOT 1: WAVEFORM + PEAK INDICATOR
    ax1 = axes[0]
    ax1.plot(time, y, linewidth=0.8, color='steelblue', alpha=0.9, label='Waveform')
    ax1.axhline(0, color='black', linestyle=':', linewidth=1.2, alpha=0.8)
    ax1.axvline(peak_time, color='red', linestyle='--', linewidth=2.5, 
                label=f'Absolute Amplitude Peak ({peak_time:.2f}s)')
    ax1.axvspan(crop_start, crop_end, alpha=0.15, color='yellow', label='3s Crop Region')
    ax1.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    ax1.set_title('Waveform and Absolute Peak Localization', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # SUBPLOT 2: F0 + RMS ENERGY
    ax2 = axes[1]
    f0_normalized = np.where(f0 > 0, f0, np.nan)
    ax2_f0 = ax2
    ax2_f0.plot(times_frame, f0_normalized, linewidth=2, color='green', 
               alpha=0.8, label='F0 (Pitch)', marker='o', markersize=3)
    ax2_f0.set_ylabel('F0 (Hz)', fontsize=11, fontweight='bold', color='green')
    ax2_f0.tick_params(axis='y', labelcolor='green')
    ax2_f0.grid(True, alpha=0.3)
    
    ax2_rms = ax2.twinx()
    ax2_rms.plot(times_frame, rms, linewidth=2, color='orange', alpha=0.7, 
                label='RMS Energy', linestyle='--', marker='s', markersize=3)
    ax2_rms.set_ylabel('RMS Energy', fontsize=11, fontweight='bold', color='orange')
    ax2_rms.tick_params(axis='y', labelcolor='orange')
    
    ax2.axvspan(crop_start, crop_end, alpha=0.15, color='yellow', label='Crop Region')
    ax2.axvline(peak_time, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
    
    lines1, labels1 = ax2_f0.get_legend_handles_labels()
    lines2, labels2 = ax2_rms.get_legend_handles_labels()
    ax2_f0.legend(lines1 + lines2 + [mpatches.Patch(facecolor='yellow', alpha=0.15, label='Crop Region')],
                 labels1 + labels2 + ['Crop Region'], loc='upper right', fontsize=9)
    
    ax2.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax2.set_title('Prosodic Contour (F0 and RMS) Analysis', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / f"02_cropping_analysis_{Path(audio_path).stem}.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[*] Saved Peak Cropping Critique Plot: {output_file}")
    plt.close()

# ============================================================================
# FUNCTION 4: INSTANTANEOUS SPECTRAL AND TEMPORAL ANALYSIS
# ============================================================================

def plot_single_frame_acoustic_features(audio_path: str, dataset_name: str = "Audio"):
    """
    Replicates instantaneous acoustic analysis visualizations:
    1. Spectral Centroid: Magnitude spectrum of a single high-energy frame with the centroid mathematically marked.
    2. Zero Crossing Rate: 300ms waveform zoom to visually inspect the zero crossings density.
    """
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    # Locate highest energy frame to guarantee voice activity analysis
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    max_energy_frame = np.argmax(rms)
    
    # Compute STFT for magnitude spectrum
    D = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    
    # ---------------------------------------------------------
    # 1. SPECTRAL CENTROID (Single Frame Magnitude Spectrum)
    # ---------------------------------------------------------
    spectrum = D[:, max_energy_frame]
    spectrum_db = librosa.amplitude_to_db(spectrum, ref=np.max)
    
    # Mathematical implementation: C = sum(f * X) / sum(X)
    centroid_hz = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
    
    # ---------------------------------------------------------
    # 2. ZERO CROSSING RATE (Waveform Zoom)
    # ---------------------------------------------------------
    center_time = librosa.frames_to_time(max_energy_frame, sr=sr, hop_length=HOP_LENGTH)
    start_time = max(0, center_time - 0.15)
    end_time = min(len(y)/sr, start_time + 0.3)
    
    # ---------------------------------------------------------
    # PLOTTING
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=150)
    fig.suptitle(f'{dataset_name} - Instantaneous Acoustic Features', fontsize=14, fontweight='bold')
    
    # PLOT 1: Spectral Centroid
    ax1 = axes[0]
    ax1.plot(freqs, spectrum_db, color='#0072BD', linewidth=1.5)
    ax1.plot(centroid_hz, 0, 'ro', markersize=10, label=f'Centroid: {centroid_hz:.0f} Hz')
    
    ax1.set_xlim([0, 3000]) 
    ax1.set_ylim([-80, 5])
    ax1.set_xlabel('Frequency (Hz)', fontsize=11)
    ax1.set_ylabel('Magnitude (dB)', fontsize=11)
    ax1.set_title('Spectral Centroid (Single Frame Analysis)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # PLOT 2: ZCR Waveform Zoom
    ax2 = axes[1]
    time_axis = np.linspace(0, len(y)/sr, len(y))
    ax2.plot(time_axis, y, color='blue', linewidth=0.8)
    
    ax2.set_xlim([start_time, end_time])
    ax2.axhline(0, color='black', linewidth=1, alpha=0.6)
    
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Amplitude', fontsize=11)
    ax2.set_title('Zero Crossing Rate (300ms Waveform Zoom)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / f"03_instantaneous_features_{Path(audio_path).stem}.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[*] Saved Instantaneous Features Plot: {output_file}")
    plt.close()

# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    print("\n" + "="*80)
    print("SER - ACOUSTIC DOMAIN KNOWLEDGE ANALYSIS INITIALIZATION")
    print("="*80)
    
    print("\n[VERIFYING DATASETS]")
    print("-" * 80)
    print(f"Data Directory:  {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    if not AUDIO_FILE_RAVDESS or not Path(AUDIO_FILE_RAVDESS).exists():
        print(f"  [X] RAVDESS audio file NOT found.")
    else:
        print(f"  [OK] RAVDESS audio file loaded.")

    if not AUDIO_FILE_IEMOCAP or not Path(AUDIO_FILE_IEMOCAP).exists():
        print(f"  [X] IEMOCAP audio file NOT found.")
    else:
        print(f"  [OK] IEMOCAP audio file loaded.")
    print("-" * 80)
    
    # STEP 0: Log Parameters
    print_stft_parameters()
    
    # STEP 1: Feature Extraction and Comparison
    print("\n[STEP 1] Generating Cross-Domain Feature Comparison Subplots...")
    print("-" * 80)
    if AUDIO_FILE_RAVDESS and AUDIO_FILE_IEMOCAP:
        try:
            plot_multi_feature_comparison(AUDIO_FILE_RAVDESS, AUDIO_FILE_IEMOCAP,
                                         label_1="RAVDESS (Clean)",
                                         label_2="IEMOCAP (Spontaneous/Noisy)")
        except Exception as e:
            print(f"  [!] Error generating comparison features: {e}")
    else:
        print("  [!] Audio files missing for comparison.")
    
    # STEP 2: Peak-Centered Cropping Analysis
    print("\n[STEP 2] Generating Peak-Centered Cropping Analysis...")
    print("-" * 80)
    if AUDIO_FILE_RAVDESS:
        plot_peak_centered_analysis(AUDIO_FILE_RAVDESS, dataset_name="RAVDESS")
    if AUDIO_FILE_IEMOCAP:
        plot_peak_centered_analysis(AUDIO_FILE_IEMOCAP, dataset_name="IEMOCAP")
        
    # STEP 3: Instantaneous Spectral and Temporal Analysis
    print("\n[STEP 3] Generating Instantaneous Spectral & Temporal Analysis...")
    print("-" * 80)
    if AUDIO_FILE_RAVDESS:
        plot_single_frame_acoustic_features(AUDIO_FILE_RAVDESS, dataset_name="RAVDESS")
    if AUDIO_FILE_IEMOCAP:
        plot_single_frame_acoustic_features(AUDIO_FILE_IEMOCAP, dataset_name="IEMOCAP")

    print("\n" + "="*80)
    print(f"✓ ANALYSIS PIPELINE COMPLETED")
    print(f"✓ All outputs successfully saved to: {OUTPUT_DIR}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()