#local path to datasets
DATASET_PATH = "data"
RAVDESS_PATH = "data/ravdess"
IEMOCAP_PATH = "data/iemocap/IEMOCAP_full_release"


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
# MelSpectrogram: hop_length=256 → frame = (48000 - 1024) / 256 + 1 ≈ 188
TIME_STEPS_RAVDESS = 188
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


# ============================================================================
# TRAINING HYPERPARAMETERS - IEMOCAP
# ============================================================================

# Basic Configuration
BATCH_SIZE_IEMOCAP = 64  # Aumentato da 32 per gradiente meno rumoroso
LEARNING_RATE_IEMOCAP = 0.0001  # Ridotto da 0.0005 per stabilità e ridurre oscillazioni
NUM_EPOCHS_IEMOCAP = 100
NUM_CLASSES_IEMOCAP = 4  # Neutral, Happy, Sad, Angry

# Audio Configuration
# Audio fissi a 3 secondi @ 16kHz = 48000 campioni
# MelSpectrogram: hop_length=256 → frame = (48000 - 1024) / 256 + 1 ≈ 188
TIME_STEPS_IEMOCAP = 188
MEL_BANDS_IEMOCAP = 128

# Model Configuration
DROPOUT_IEMOCAP = 0.4  # Ridotto per preservare feature sottili (es. Sad)

# Augmentation Configuration
SPEC_FREQ_MASK_IEMOCAP = 30  # Aumentato da 12: costringe a imparare differenze sottili tra Angry/Happy
SPEC_TIME_MASK_IEMOCAP = 15  # Ridotto per preservare feature sottili

# Class Weights Configuration (Neutral, Happy, Sad, Angry)
CLASS_WEIGHTS_IEMOCAP = [1.0, 1.0, 1.5, 1.0]  # Neutral 1.0 evita falsi Happy, Sad 2.0 per aiutarlo

# SWA Configuration
SWA_START_EPOCH_IEMOCAP = 15  # Inizia SWA dopo 15 epoche
SWA_LR_IEMOCAP = 0.0001  # Learning rate costante per SWA

# Regularization
WEIGHT_DECAY_IEMOCAP = 0.001  # Aumentato da 0.0001 per ridurre overfitting e oscillazioni


# ============================================================================
# KNOWLEDGE DISTILLATION & NOISY STUDENT TRAINING - IEMOCAP
# ============================================================================
# Configurazione per il training di Student Network usando Teacher Network
# (RAVDESS model come teacher) su IEMOCAP dataset
#
# Strategia: Knowledge Distillation + Noisy Student
#   1. Teacher (RAVDESS best_swa_model) genera soft targets su IEMOCAP
#   2. Student (rete vuota) impara da:
#      - Hard labels (ground truth)
#      - Soft labels (probabilità dal teacher tramite KL divergence)
#   3. Optional: Student diventa nuovo teacher (iterativa)
# ============================================================================

# ---- TEMPERATURE SCALING ----
# Parametro per il soft target generation (softmax con temperatura)
# Temperatura BASSA (T < 1):    Output probabilities concentrate on argmax → harder targets
# Temperatura ALTA (T > 1):     Output probabilities più smooth → softer targets (più informativo)
# Tipicamente: T in [2.0, 10.0]
TEMPERATURE = 4.0  # 🔥 Soft targets moderatamente lisci (default: 4.0, range: 2-8)


# ---- LOSS FUNCTION WEIGHTING ----
# La loss totale è: Loss = ALPHA_CE * CE_loss + ALPHA_KL * KL_loss
# Dove:
#   - CE_loss:  CrossEntropyLoss tra student predictions e hard labels (ground truth)
#   - KL_loss:  Kullback-Leibler divergence tra student e teacher (soft targets)
#
# Trade-off:
#   ALPHA_CE alto (0.8-1.0):  Focalizza su ground truth → performante su train set, rischio overfitting
#   ALPHA_KL alto (0.3-0.5):  Focalizza su teacher knowledge → migliore generalizzazione
#
# Consigliato: ALPHA_CE + ALPHA_KL = 1.0 (somma a 1 per normalizzazione)
ALPHA_CE = 0.7   # 🎯 Peso per CrossEntropyLoss (hard labels) - default: 0.7
ALPHA_KL = 0.3   # 📚 Peso per KL Divergence (soft labels dal teacher) - default: 0.3


# ---- NOISY STUDENT NOISE INJECTION ----
# Tecnica: Aggiungere rumore ai soft targets per rendere il training più difficile
# (forcing the student a imparare feature più robuste)
# 
# Noise types:
#   1. Gaussian Noise: η ~ N(0, σ²) → rumore additivo nei logit del teacher
#   2. Label Smoothing: p_smooth = (1-ε) * p + ε/K → distribuzione più uniforme
#   3. DropOut aggresivo: aumentare dropout durante student training
#
# Note: 
#   - Noise moderato migliora generalization (regolarizzazione)
#   - Noise eccessivo → soft targets poco informativi
NOISE_VARIANCE = 0.01         # 🔊 Varianza rumore gaussiano sui logit - default: 0.01 (range: 0.005-0.05)
LABEL_SMOOTHING = 0.1         # 📊 Label smoothing per soft targets - default: 0.1 (range: 0.05-0.2)

# ---- HARD PSEUDO-LABELS CONFIGURATION ----
CONFIDENCE_THRESHOLD = 0.7  # 🎯 Soglia di confidenza per accettare una pseudo-label dal Teacher
ADDITIVE_NOISE_SNR_MIN = 10 # 🔊 SNR Minimo per Additive Gaussian Noise sulla waveform
ADDITIVE_NOISE_SNR_MAX = 20 # 🔊 SNR Massimo per Additive Gaussian Noise sulla waveform


# ---- STUDENT TRAINING HYPERPARAMETERS ----
# Iperparametri specifici per il training dello student network
# (diversi da teacher per accelerare convergenza e sfruttare prior knowledge)
#
# Motivation:
#   - Meno epoche: Student inizia da knowledge del teacher → convergenza rapida
#   - Learning rate simile: Stabilità, evitare divergenza quando segue teacher soft targets
#   - Batch size: Uguale al teacher per consistency nelle feature estratte
NUM_EPOCHS_STUDENT_IEMOCAP = 80    # 🚀 Epoche per student (meno del teacher 100) - default: 80
BATCH_SIZE_STUDENT_IEMOCAP = 64    # 📦 Batch size per student (uguale al teacher) - default: 64
LEARNING_RATE_STUDENT_IEMOCAP = 0.0001  # 🎚️ Learning rate per student - default: 0.0001
WEIGHT_DECAY_STUDENT_IEMOCAP = 0.001    # 🛡️ L2 regularization per student - default: 0.001
DROPOUT_STUDENT_IEMOCAP = 0.5       # 🎲 Dropout student (più aggressivo del teacher 0.4) - default: 0.5


# ---- SWA CONFIGURATION FOR STUDENT ----
# Stochastic Weight Averaging per il student network
# (stesso principio del teacher: media pesi per migliore generalizzazione)
SWA_START_EPOCH_STUDENT = 10   # 📈 Inizio SWA dopo N epoche - default: 10
SWA_LR_STUDENT = 0.0001        # 🔄 Learning rate during SWA - default: 0.0001


# ---- EARLY STOPPING (STUDENT) ----
# Fermare il training se la validation loss non migliora per N epoche
EARLY_STOPPING_PATIENCE_STUDENT = 8  # ⏸️ Patience per early stopping - default: 8


# ---- ITERATIVE TEACHER-STUDENT ----
# Opzionale: Fare più iterazioni dove Student diventa nuovo Teacher
# (self-training style: pseudo-labeling circolare)
#
# Processo:
#   Iteration 1: T0 (RAVDESS) → T1 (trained student)
#   Iteration 2: T1 → T2 (nuovo student trained con T1 soft labels)
#   Iteration 3: T2 → T3, ...
#
# Trade-off: Ogni iterazione può amplificare errori del teacher
NUM_ITERATIONS_NOISY_STUDENT = 1    # 🔁 Numero di iterazioni (1 = no iterazione) - default: 1 (range: 1-3)