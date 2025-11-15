
# Dual-Stream Age-Aware Iris Stress Detection Model
## Training Documentation

**Generated:** 2025-11-11 22:04:48

---

## Model Architecture

### Overview
- **Type:** Dual-Stream Convolutional Neural Network
- **Input Streams:**
  1. **Pupil Stream:** 224×224×5 channels (RGB only - channels 3-4 zeroed)
  2. **Iris Stream:** 224×224×5 channels (RGB + Canny + BlackHat)
  3. **Age Features:** 8 age groups (one-hot encoding)
  4. **Iris Ring Count:** Scalar tension ring count from pattern analysis

- **Key Innovation:** Age features are injected into pupil stream BEFORE fusion
  - Biologically accurate: Age affects pupil dilation, not iris tension rings
  - Explicit modeling: Fusion compares "Pupil+Age context" vs "Iris patterns"

### Novel Components
1. **WeightedFeatureFusion Layer** ⭐ (MOST NOVEL)
   - Learnable α parameter computed per-sample by gating network
   - Formula: `output = α * iris_features + (1-α) * pupil_age_features`
   - Adaptive fusion: Model learns which stream to trust for each sample

2. **EdgeAttentionModule**
   - Spatial attention for tension ring boundaries and pupil edges
   - Input: Edge channel + RGB → Output: 16 channels

3. **FeatureAttentionModule**
   - Channel attention for texture patterns (BlackHat morphology)
   - Input: Texture channel + RGB → Output: 16 channels

4. **5-Channel Feature Extraction**
   - Channel 0-2: RGB (color information)
   - Channel 3: Canny edges (boundaries detection)
   - Channel 4: BlackHat morphology (dark structure detection)

### Total Parameters
- **486,772** parameters

---

## Dataset Information

### Distribution
- **Pupil Dataset:** 2,504 images
  - Normal: 1,706
  - Stressed: 798
  - Ratio: 2.14:1

- **Iris Dataset:** 1,191 images
  - Normal: 885
  - Stressed: 306

### Data Split
- **Training:** 1,388 samples (70%)
- **Validation:** 850 samples (20%)
- **Test:** 266 samples (10%)

### Optimized Balancing Strategy (70-20-10 Split)
**Original Imbalance:** 945:443 = 2.13:1

**Solution:**
1. **Step 1 (Pupil Augmentation):** 1.5x multiplication of stressed samples
   - Result: 945:664
   
2. **Step 2 (Iris Augmentation):** 2x multiplication of stressed iris images
   - Random iris pairing prevents memorization
   
3. **Step 3 (Focal Loss):** Alpha = 0.5, Gamma = 2.0
   - Handles class imbalance without explicit class weights
   - Focuses on hard-to-classify samples

**Benefits:**
- ✅ Stratified split ensures balanced validation/test sets
- ✅ Reduced augmentation (from 3x to 2x iris) prevents overfitting
- ✅ Focal Loss handles imbalance elegantly
- ✅ Warmup learning rate for stable convergence

---

## Training Configuration

- **Batch Size:** 32
- **Epochs:** 50
- **Initial Learning Rate:** 1e-06 (warmup start)
- **Peak Learning Rate:** 5e-05 (after warmup)
- **Warmup Epochs:** 5
- **Optimizer:** Adam
- **Loss Function:** Focal Loss (α=0.5, γ=2.0)
- **Split Strategy:** 70-20-10 Stratified (class-balanced validation/test)

### Callbacks
- **ModelCheckpoint:** Saves best model (monitor: val_auc_pr)
- **EarlyStopping:** Patience = 15 epochs
- **Smart Stop:** Stops if AUC ≥ 0.95 for 5 consecutive epochs
- **ReduceLROnPlateau:** Reduces LR by 0.5x every 5 epochs (patience=5)
- **AlphaMonitor:** Tracks learnable fusion parameter evolution

---

## Test Set Results

- **Test Loss:** 0.1206
- **Test Accuracy:** 0.9412
- **Test AUC (ROC):** 0.9850
- **Test AUC-PR:** 0.9815
- **Optimal Threshold:** 0.4498

### Alpha Analysis (Learnable Fusion)

- **Mean Alpha:** 0.8171
- **Std Dev:** 0.0774
- **Interpretation:** Iris-weighted

  → Model trusts **IRIS STREAM** more (tension ring patterns)

---

## Files Generated

1. **Model File:** `best_dual_stream_age_aware_model.keras`
2. **Training History:** `training_history.csv`
3. **Test Predictions:** `test_predictions.csv`
4. **Metadata:** `training_metadata.json`
5. **Visualizations:**
   - `training_history.png` (6-panel: loss, accuracy, AUC, AUC-PR, alpha evolution)
   - `stream_importance_evolution.png` (epoch-by-epoch stream split)
   - `evaluation_metrics.png` (confusion matrix, ROC, PR curves)
   - `alpha_analysis.png` (alpha distribution, by class, vs confidence)

---

## Integration with Flask Backend

### Loading the Model
```python
import tensorflow as tf
from tensorflow import keras

# Load model
model = keras.models.load_model('best_dual_stream_age_aware_model.keras')

# Model expects 3 inputs:
# 1. pupil_input: (batch, 224, 224, 5)
# 2. iris_input: (batch, 224, 224, 5)
# 3. age_input: (batch, 8)
```

### Input Preprocessing
1. Load pupil and iris images (640×480 or varied dimensions)
2. Resize to 224×224 with aspect ratio preservation
3. Extract 5-channel features (RGB + Canny + BlackHat)
4. Zero out channels 3-4 for pupil stream
5. Create age one-hot encoding (8 groups)
6. Normalize to [0, 1] range

### Prediction
```python
prediction = model.predict({
    'pupil_input': pupil_array,
    'iris_input': iris_array,
    'age_input': age_array
})

stress_probability = prediction[0][0]
is_stressed = stress_probability > 0.5
```

---

## Key Achievements

✅ **Novel Architecture:** Learnable α fusion with age-aware design
✅ **Optimized Training:** 70-20-10 stratified split with reduced augmentation
✅ **Stable Convergence:** Warmup LR + Focal Loss for balanced learning
✅ **High Performance:** AUC = 0.9850, AUC-PR = 0.9815
✅ **Biologically Accurate:** Age affects pupil, not iris (explicit modeling)
✅ **Production Ready:** Integrated with Flask backend
✅ **Smart Early Stopping:** Stops at optimal performance (AUC ≥ 0.95 for 5 epochs)

---

**End of Documentation**
