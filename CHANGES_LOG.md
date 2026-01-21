# Training Changes Log

## Run 1: Jan 21, 2026 - Optimized Hyperparameters

### Applied Changes:
1. **Class Weights**: `[1.0, 1.0, 2.0, 1.0]` (Sad weight = 2.0)
2. **SpecAugment Freq Mask**: `30` (increased from 12)
3. **Model Block4**: Channels reduced to 128 (from 256)
4. **Learning Rate**: Keep `0.0001`
5. **Batch Size**: Keep `64`
6. **Weight Decay**: Keep `0.001`
7. **Data Augmentation**: 
   - Gaussian Noise: 0.001-0.005 (50%)
   - Time Shift: max 0.1s (50%)
   - Amplitude Gain: 0.8-1.2 (50%)
8. **Dropout**: 0.3 (final layer)

### Architecture Changes:
- Block 1: 1 → 128 channels (same)
- Block 2: 128 → 128 channels (same)
- Block 3: 128 → 256 channels (same)
- Block 4: 256 → 128 channels (**REDUCED** from 256)
- LSTM Input Size: 1024 (from 2048)
- Projection: 1024 → 128

### Expected Impact:
- ✅ Stronger focus on Sad emotion (weight 2.0)
- ✅ More aggressive spectrogram augmentation (freq_mask=30)
- ✅ Further parameter reduction (Block4 128 channels)
- ✅ Better generalization on small dataset (1440 samples)

### Files Modified:
- `config.py`: CLASS_WEIGHTS, SPEC_FREQ_MASK
- `models/crnn_bilstm.py`: Block4 channels + lstm_input_size
- `models/crnn_bigru.py`: Block4 channels + gru_input_size

### Command to Run:
```bash
python train.py --model CRNN_BiLSTM
```

### Rollback Instructions (if needed):
```python
# config.py
CLASS_WEIGHTS_RAVDESS = [1.0, 1.0, 1.5, 1.0]
SPEC_FREQ_MASK_RAVDESS = 12

# models/crnn_bilstm.py & crnn_bigru.py
# Block4: Conv2d(256, 256, ...)  # Change 256 to 128 back to 256
# lstm_input_size = 256 * 8  # Change 128 to 256
```

---

## Previous Runs:
- Focal Loss: REMOVED (peggiorato le prestazioni)
- Initial optimization: Batch size 64, LR 0.0001, Weight decay 0.001
