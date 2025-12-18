"""
Utility functions for visualizing audio samples and their features.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def visualize_samples(dataloader, num_samples=4):
    """
    Visualize log spectrograms and information for samples from the dataloader.
    
    Args:
        dataloader: PyTorch DataLoader with audio samples
        num_samples: Number of samples to visualize (default: 4)
    """
    
    # Get first batch from dataloader
    batch = next(iter(dataloader))
    
    # Limit to num_samples
    num_to_show = min(num_samples, len(batch['audio_features']))
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_to_show, 1, figsize=(14, 4 * num_to_show))
    
    # Handle case when there's only 1 sample
    if num_to_show == 1:
        axes = [axes]
    
    for idx in range(num_to_show):
        ax = axes[idx]
        
        # Get sample data
        log_spec = batch['audio_features'][idx].numpy()  # (1, freq_bins, time_frames)
        label = batch['label'][idx]
        speaker_id = batch['speaker_id'][idx]
        session_id = batch['session_id'][idx]
        
        # Remove channel dimension for visualization
        log_spec = log_spec[0]  # (freq_bins, time_frames)
        
        # Plot spectrogram
        im = ax.imshow(log_spec, aspect='auto', origin='lower', cmap='viridis')
        ax.set_ylabel('Frequency Bin')
        ax.set_xlabel('Time Frame')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, format='%+2.0f dB')
        cbar.set_label('Magnitude (dB)')
        
        # Add title with information
        title = f"Emotion: {label} | Speaker: {speaker_id} | Session: {session_id} | Shape: {log_spec.shape}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def print_batch_info(batch):
    """
    Print information about a batch from the dataloader.
    
    Args:
        batch: Dictionary containing batch data
    """
    print("=" * 80)
    print("BATCH INFORMATION")
    print("=" * 80)
    print(f"Batch size: {len(batch['label'])}")
    print(f"Audio features shape: {batch['audio_features'].shape}")  # (batch_size, channels, freq_bins, time_frames)
    print(f"\nSample details:")
    print("-" * 80)
    
    for idx in range(len(batch['label'])):
        print(f"Sample {idx + 1}:")
        print(f"  - Emotion Label: {batch['label'][idx]}")
        print(f"  - Speaker ID: {batch['speaker_id'][idx]}")
        print(f"  - Session ID: {batch['session_id'][idx]}")
        print(f"  - Log Spectrogram shape: {batch['audio_features'][idx].shape}")
    
    print("=" * 80)
