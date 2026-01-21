#local path to datasets
DATASET_PATH = "data"
RAVDESS_PATH = "data/ravdess"
IEMOCAP_PATH = "data/iemocap"


#colab path to datasets 
COLAB_DATASET_PATH = "/content/data"
COLAB_RAVDESS_PATH = "/kaggle/input/ravdess-emotional-speech-audio"
COLAB_IEMOCAP_PATH = "/kaggle/input/iemocapfullrelease/IEMOCAP_full_release"


# ============================================================================
# TRAINING HYPERPARAMETERS - RAVDESS
# ============================================================================

# Basic Configuration
BATCH_SIZE_RAVDESS = 64  # Aumentato da 32 per gradiente meno rumoroso
LEARNING_RATE_RAVDESS = 0.0001  # Ridotto da 0.0005 per stabilità e ridurre oscillazioni
NUM_EPOCHS_RAVDESS = 100
NUM_CLASSES_RAVDESS = 4  # Neutral, Happy, Sad, Angry

# Audio Configuration
# Audio fissi a 3 secondi @ 16kHz = 48000 campioni
# MelSpectrogram: hop_length=512 → frame = (48000 - 2048) / 512 + 1 ≈ 90
TIME_STEPS_RAVDESS = 90
MEL_BANDS_RAVDESS = 128

# Model Configuration
DROPOUT_RAVDESS = 0.4  # Ridotto per preservare feature sottili (es. Sad)

# Augmentation Configuration
SPEC_FREQ_MASK_RAVDESS = 30  # Aumentato da 12: costringe a imparare differenze sottili tra Angry/Happy
SPEC_TIME_MASK_RAVDESS = 15  # Ridotto per preservare feature sottili

# Class Weights Configuration (Neutral, Happy, Sad, Angry)
CLASS_WEIGHTS_RAVDESS = [1.0, 1.0, 2.0, 1.0]  # Neutral 1.0 evita falsi Happy, Sad 2.0 per aiutarlo

# SWA Configuration
SWA_START_EPOCH_RAVDESS = 15  # Inizia SWA dopo 15 epoche
SWA_LR_RAVDESS = 0.0001  # Learning rate costante per SWA

# Regularization
WEIGHT_DECAY_RAVDESS = 0.001  # Aumentato da 0.0001 per ridurre overfitting e oscillazioni
