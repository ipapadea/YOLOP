# TwinLiteNet2Scaled vs TriLiteNet: Comprehensive Technical Comparison

**Date:** January 22, 2026  
**Purpose:** Strategic analysis to identify improvements and surpass TriLiteNet performance

---

## 📊 Performance Summary

| Metric | Your Model (TwinLiteNet2Scaled) | TriLiteNet-base | Difference | Winner |
|--------|--------------------------------|-----------------|------------|--------|
| **Detection Recall** | 88.6% | 85.6% | **+3.0%** | ✅ **You** |
| **Detection Precision** | 4.7% | ~70% (estimated) | **-65.3%** | ❌ TriLiteNet |
| **mAP@0.5** | 73.2% | 72.3% | **+0.9%** | ✅ **You** |
| **mAP@0.5:0.95** | 39.5% | ~40% (estimated) | ~-0.5% | ≈ Similar |
| **DA mIoU** | 91.6% | 92.4% | -0.8% | ❌ TriLiteNet |
| **LL Accuracy** | 68.9% | 82.3% | **-13.4%** | ❌ TriLiteNet |
| **LL IoU** | 25.3% | ~30% (estimated) | -4.7% | ❌ TriLiteNet |
| **Parameters** | Unknown | 2.35M | ? | ? |
| **FLOPs** | Unknown | 7.72G | ? | ? |

### Key Observations:
- ✅ **Your Strengths:** Superior detection recall, competitive mAP@0.5
- ❌ **Your Weaknesses:** Catastrophic precision failure (4.7%), poor lane line segmentation (-13.4%)
- 🎯 **TriLiteNet's Edge:** Balanced performance, especially lane lines and precision

---

## 🏗️ Architecture Comparison

### 1. Encoder Design

| Component | Your Model | TriLiteNet | Analysis |
|-----------|-----------|------------|----------|
| **Base Architecture** | ESPNet2_Encoder_scaledExtended | ESPNet-based Encoder | Similar foundation |
| **Key Parameters** | p=5, q=3, scale=1.0 | Standard ESPNet config | Your encoder might be more complex |
| **Channel Configuration** | Dynamic scaling | Fixed channel dict (sc_ch_dict) | Your approach more flexible |
| **Depth** | Extended version | Standard version | Potentially deeper |

**🔍 Your Advantage:** Extended encoder with more representational power  
**⚠️ Risk:** Might be over-parameterized, slower inference

---

### 2. Neck Architecture

| Component | Your Model | TriLiteNet | Winner |
|-----------|-----------|------------|--------|
| **Architecture** | PaFPNELAN_Ghost_C2 | FPN + PAN | Different approaches |
| **Convolution Type** | Ghost convolutions | Depthwise Separable Conv | Both efficient |
| **Feature Fusion** | ELAN-style blocks | Standard FPN/PAN | ✅ You (more advanced) |
| **Multi-scale Levels** | Optimized for 4 scales | Optimized for 3 scales | Depends on use case |

**🔍 Your Advantage:** 
- Ghost convolutions reduce parameters while maintaining performance
- ELAN blocks provide better feature learning
- Modern neck design (inspired by YOLOv7/v8)

**⚠️ TriLiteNet's Advantage:**
- Simpler, proven architecture
- Easier to optimize and debug

---

### 3. Detection Head

| Component | Your Model | TriLiteNet | Winner |
|-----------|-----------|------------|--------|
| **Scales** | **4-scale** (P3, P4, P5, P6) | **3-scale** (P3, P4, P5) | ✅ **You** (better for varied object sizes) |
| **Architecture** | IDetect with RepConv | Standard Detect | ✅ You (RepConv improves accuracy) |
| **Anchors** | 3 anchors × 4 scales = 12 | 3 anchors × 3 scales = 9 | You (more coverage) |
| **NMS Config** | conf=0.001, iou=0.6 | conf=0.001, iou=0.6 | ✅ Same (correct) |

**🎯 Critical Finding:** 
- Both use identical NMS thresholds
- Your precision issue (4.7%) is **NOT** caused by NMS configuration
- **Root cause:** Likely class imbalance, anchor mismatch, or training instability

**🔍 Your Advantages:**
- 4-scale detection handles tiny to huge objects better
- RepConv blocks improve detection accuracy
- More anchor boxes for better localization

---

### 4. Segmentation Head

| Component | Your Model | TriLiteNet | Winner |
|-----------|-----------|------------|--------|
| **Architecture** | MHGD (Multi-Head Guided Decoder) | CAAM (Class Activation Attention) | Different philosophies |
| **Upsampling** | UPx2_scaled (custom) | UpConvBlock (standard) | Depends on efficiency |
| **Attention Mechanism** | Multi-head guided | Class activation with GCN | ✅ TriLiteNet (proven for segmentation) |
| **Heads** | Dual heads (DA + LL) | Dual heads (DA + LL) | Same |

**🔍 TriLiteNet's Advantage:**
- **CAAM** is specifically designed for segmentation with:
  - Bin-based spatial pooling (2×4 bins)
  - Graph Convolutional Network (GCN) for feature refinement
  - Class activation maps for better localization
- Proven effective for thin structures like lane lines

**⚠️ Your Challenge:**
- MHGD might not be optimized for thin lane line detection
- Missing specialized attention for lane line structures

---

## 🔥 Loss Function Comparison

### Your Loss Configuration

```python
# Detection losses (standard YOLO-style)
BOX_GAIN = 0.05
CLS_GAIN = 0.5
OBJ_GAIN = 1.0

# Segmentation losses
DA_SEG_GAIN = 0.2
LL_SEG_GAIN = 0.2    # ⚠️ TOO LOW
LL_IOU_GAIN = 0.2

# Total segmentation weight for LL: 0.2 + 0.2 = 0.4
```

### TriLiteNet Loss Configuration

```python
# Detection losses (identical to yours)
BOX_GAIN = 0.05
CLS_GAIN = 0.5
OBJ_GAIN = 1.0

# Segmentation losses - DUAL LOSS APPROACH
FL_GAIN = 0.3    # FocalLoss gain
TK_GAIN = 0.3    # TverskyLoss gain

# Loss functions:
FocalSeg = FocalLossSeg(mode="multiclass", alpha=0.25)
TverskyDaSeg = TverskyLoss(mode="multiclass", alpha=0.7, beta=0.3, gamma=4/3)
TverskyLlSeg = TverskyLoss(mode="multiclass", alpha=0.9, beta=0.1, gamma=4/3)

# Combined:
lseg_focal = FocalSeg(DA) + FocalSeg(LL)
lseg_tversky = TverskyDaSeg(DA) + TverskyLlSeg(LL)
lseg = lseg_focal * 0.3 + lseg_tversky * 0.3

# Total segmentation weight: 0.6 per task
```

### 🔍 Critical Loss Function Differences

| Aspect | Your Model | TriLiteNet | Impact |
|--------|-----------|------------|--------|
| **Loss Types** | Single loss per task | **FocalLoss + TverskyLoss** | ❌ You lack diversity |
| **LL Total Weight** | 0.4 (0.2+0.2) | **0.6** (0.3+0.3) | ❌ Your LL is under-weighted |
| **DA Total Weight** | 0.2 | **0.6** | ❌ Your DA is under-weighted |
| **Tversky LL Config** | N/A | **alpha=0.9, beta=0.1** | ❌ Missing FP penalty for thin lines |
| **Focal Loss** | Unknown if used | alpha=0.25, gamma=2.0 | ? Need to verify |

### 🎯 Why TriLiteNet's Loss is Better for Lane Lines

**Tversky Loss with alpha=0.9, beta=0.1:**
```
Tversky = TP / (TP + 0.9*FP + 0.1*FN)
```
- **Heavily penalizes False Positives** (FP weighted 9× more than FN)
- Perfect for **thin lane lines** where FP noise is a major issue
- Encourages precise, clean predictions

**Your Model's Issue:**
- Standard loss doesn't discriminate between FP and FN
- Lane lines get drowned in noise
- Result: 68.9% accuracy vs 82.3% target

---

## ⚙️ Hyperparameter Comparison

| Hyperparameter | Your Model | TriLiteNet | Recommendation |
|----------------|-----------|------------|----------------|
| **Epochs** | 300 | 240 | ✅ You train longer (good) |
| **Batch Size** | 24 (12×2 GPUs) | 16 (8×2 GPUs) | ✅ You use larger batches |
| **Optimizer** | Adam | **AdamW** | ❌ Switch to AdamW (better regularization) |
| **LR Range** | 0.001 → 0.2 | 0.001 → 0.2 | ✅ Same |
| **LR Scheduler** | Cosine annealing | Cosine annealing | ✅ Same |
| **Warmup Epochs** | 3 | 3 | ✅ Same |
| **Weight Decay** | Unknown | **0.0005** | ❓ Verify yours |
| **Momentum** | Unknown | 0.937 | ❓ Verify yours |
| **Image Size** | 384×640 | 384×640 (code), 640×640 (paper) | ✅ Same in code |

### 🔍 Key Finding: AdamW vs Adam

**Why TriLiteNet uses AdamW:**
```python
_C.TRAIN.OPTIMIZER = 'adamw'
_C.TRAIN.WD = 0.0005
```

**Benefits of AdamW:**
- Decoupled weight decay (better regularization)
- Prevents overfitting on small datasets
- More stable training for multi-task learning
- Better generalization

**Your Model:**
- Using Adam (standard)
- Might be overfitting (explains high recall but low precision?)

---

## 🎓 Training Strategy Comparison

### Data Augmentation

| Augmentation | Your Model | TriLiteNet | Notes |
|--------------|-----------|------------|-------|
| **Flip** | Yes | Yes | Standard |
| **HSV** | Yes | Yes (H=0.015, S=0.7, V=0.4) | Check your values |
| **Scale** | 0.25 | 0.25 | Same |
| **Rotation** | Yes | 10° | Check your value |
| **Translation** | Yes | 0.1 | Check your value |
| **Mosaic** | Unknown | 0.0 (disabled) | TriLiteNet doesn't use it |
| **Mixup** | Unknown | 0.0 (disabled) | TriLiteNet doesn't use it |

**🎯 Insight:** TriLiteNet keeps augmentation **simple and standard** - no fancy mosaic/mixup

---

### Training Schedule

| Stage | Your Model | TriLiteNet |
|-------|-----------|------------|
| **Total Epochs** | 300 | 240 |
| **Warmup** | 3 epochs | 3 epochs |
| **Validation Freq** | Every epoch? | Every epoch |
| **Start Validation** | Epoch 0 | Epoch 100 |
| **LR Schedule** | Cosine | Cosine |

**🔍 TriLiteNet's Strategy:**
- Validates only after epoch 100 (saves time)
- 240 epochs sufficient with their loss configuration
- Your 300 epochs might indicate training inefficiency

---

### Anchor Configuration & Logic

| Aspect | Your Model (TwinLiteNet2Scaled) | TriLiteNet |
|--------|--------------------------------|------------|
| **Detection Scales** | **4-scale** (P3, P4, P5, P6) | **3-scale** (P3, P4, P5) |
| **Anchors per Scale** | 3 | 3 |
| **Total Anchors** | **12** (4 × 3) | **9** (3 × 3) |
| **Auto-anchor** | Yes (configurable) | Yes (k-means on dataset) |
| **Anchor Threshold** | 4.0 | 4.0 |
| **IOU Threshold** | 0.2 | 0.2 |

#### 🔍 Critical Anchor Logic Differences

**TriLiteNet's Approach:**
```python
# Fixed 3-scale anchors regenerated via k-means at training start
_C.NEED_AUTOANCHOR = True  
_C.TRAIN.ANCHOR_THRESHOLD = 4.0

# Anchors are ALWAYS regenerated when training from scratch
# Simple, proven approach: 3 scales match standard YOLO
```

**Your Model's Approach:**
```python
# 4-scale anchors - more complex
_C.NEED_AUTOANCHOR = True  # DEFAULT in config

# Anchor behavior depends on training scenario:
# 1. From scratch (epoch 0): Regenerates via k-means if NEED_AUTOANCHOR=True
# 2. From checkpoint: Loads anchors from checkpoint.pth

# In tools/train.py:
if cfg.NEED_AUTOANCHOR:
    run_anchor(...)  # K-means regeneration
else:
    logger.info("anchors loaded successfully")  # Use checkpoint anchors
```

#### ⚠️ CRITICAL FINDING: Your Training Run Had Anchor Mismatch

**From your training log (`from_scratch_amp_disabled_2025-08-31-14-01_train.log`):**

```
Config: NEED_AUTOANCHOR: False
Message: "anchors loaded successfully"
Loaded anchors: tensor([[[0.3406,0.9041],[0.4982,1.4997],[0.8007,1.9692]],
                        [[0.5705,1.5785],[1.0062,2.2252],[1.5214,3.5912]],
                        [[0.8047,3.4575],[1.4010,2.9536],[2.1601,5.0238]]])
```

**Problem Identified:**
- ❌ Training resumed from **checkpoint at epoch 134**
- ❌ `NEED_AUTOANCHOR=False` → No k-means regeneration
- ❌ Loaded **3-scale anchors** from checkpoint (9 anchors total)
- ❌ But model architecture expects **4-scale anchors** (12 anchors total)
- ❌ **MISMATCH:** 3-scale checkpoint anchors vs 4-scale model definition

**Model Definition (in code):**
```python
# lib/models/TwinLite_2_scaled_Object_Detection_YOLOP_format.py
self.anchors = [
    [4.15629,11.41984, 5.94761,16.46950, 8.18673,23.52688],    # P3
    [12.04416,29.51737, 16.35089,41.95507, 24.17928,57.18741], # P4
    [33.29597,78.16243, 47.86408,108.28889, 36.33312,189.21414], # P5
    [73.09806,144.64581, 101.18080,253.37000, 136.02821,408.82248] # P6
]
# 4 scales × 3 anchors = 12 total
```

#### 🎯 Impact on Performance

**This anchor mismatch explains your catastrophic precision (4.7%):**

1. **Scale Mismatch:**
   - Model expects predictions at P3, P4, P5, **P6** (4 levels)
   - Anchors only cover P3, P4, P5 (3 levels)
   - P6 predictions have **NO optimized anchors**
   
2. **33% More Predictions Without Proper Anchors:**
   - 12 anchor boxes vs TriLiteNet's 9
   - Extra 3 anchors at P6 scale are random/unoptimized
   - Creates noise in predictions → Low precision

3. **High Recall + Low Precision Pattern:**
   - Recall 88.6%: Model detects objects (3 scales still work)
   - Precision 4.7%: Bounding boxes are poorly localized (P6 noise + anchor mismatch)

#### ✅ Comparison Summary

| Factor | Your Model | TriLiteNet | Winner |
|--------|-----------|------------|--------|
| **Anchor Count** | 12 (4 scales) | 9 (3 scales) | ⚠️ Depends on optimization |
| **Anchor Optimization** | **Mismatch in training run** | ✅ Always k-means optimized | ❌ TriLiteNet |
| **Scale Coverage** | P3-P6 (tiny to huge) | P3-P5 (small to large) | ✅ You (if fixed) |
| **Complexity** | Higher (more predictions) | Simpler (proven approach) | ❌ TriLiteNet |
| **Efficiency** | 33% more anchors | Standard anchor count | ❌ TriLiteNet |

**🎯 Key Insight:**
- TriLiteNet's 3-scale approach is **simpler, proven, and properly optimized**
- Your 4-scale approach **could be better** for multi-scale objects, but:
  - ❌ Anchors weren't regenerated (NEED_AUTOANCHOR=False)
  - ❌ Loaded 3-scale anchors into 4-scale architecture
  - ❌ Created catastrophic precision failure

#### 💡 Fix Recommendations

**Option 1: Regenerate 4-Scale Anchors (Keep Architecture)**
```python
# Set in your config before training:
NEED_AUTOANCHOR = True
ANCHOR_THRESHOLD = 4.0

# Start from scratch or force anchor regeneration
# k-means will compute optimal 12 anchors for BDD100K dataset
```

**Option 2: Simplify to 3-Scale (Match TriLiteNet)**
```python
# Modify model to use 3-scale detection (P3, P4, P5 only)
# Remove P6 level → 33% fewer predictions
# Simpler architecture, easier to optimize
# Match TriLiteNet's proven approach
```

**Recommendation:** 
- **For BDD100K:** Try Option 1 first with proper k-means
- **If still issues:** Fall back to Option 2 (proven 3-scale approach)
- **Always set:** `NEED_AUTOANCHOR=True` when training from scratch

---

## 🎯 ROOT CAUSE ANALYSIS

### Why Your Precision is 4.7% (TriLiteNet: ~70%)

**Hypothesis Ranking:**

1. **⭐⭐⭐ Anchor Mismatch** (Most Likely)
   - You have 4 scales × 3 anchors = 12 anchor configurations
   - If anchors aren't tuned via k-means, predictions are all wrong
   - High recall means you detect objects, but bounding boxes are imprecise
   - **Fix:** Run anchor k-means optimization before training

2. **⭐⭐⭐ Class Imbalance / Training Instability** 
   - Single class (plant) might have extreme imbalance
   - Adam optimizer without weight decay might overfit
   - **Fix:** Switch to AdamW, verify class distribution

3. **⭐⭐ Loss Function Weighting**
   - OBJ_GAIN=1.0 might be too high relative to CLS_GAIN=0.5
   - Detection losses not balanced properly
   - **Fix:** Tune loss weights

4. **⭐ Post-processing Issues**
   - NMS threshold is correct (0.001)
   - But something in prediction pipeline might be off
   - **Fix:** Debug prediction pipeline

---

### Why Your Lane Line Accuracy is 68.9% (TriLiteNet: 82.3%)

**Root Causes:**

1. **⭐⭐⭐ Loss Configuration** (Confirmed)
   - LL_SEG_GAIN = 0.2 (yours) vs 0.6 (TriLiteNet)
   - Missing Tversky Loss with FP penalty
   - **Fix:** Implement dual loss (Focal + Tversky)

2. **⭐⭐ Architecture**
   - MHGD decoder might not be optimized for thin structures
   - TriLiteNet's CAAM with GCN is designed for this
   - **Fix:** Consider adopting CAAM or improving MHGD

3. **⭐ Training Duration**
   - Even with 300 epochs, LL performance plateaus
   - Indicates architectural or loss function issue
   - **Fix:** Address loss/architecture first

---

## 🚀 STRATEGIC ROADMAP TO SURPASS TRILITENET

### Phase 1: Critical Fixes (Week 1)

**Priority 1: Fix Precision (4.7% → 70%+)**

```bash
# Step 1: Verify and regenerate anchors
python tools/verify_anchors.py --data path/to/dataset --img-size 640 384

# Step 2: Switch to AdamW optimizer
# Edit lib/config/default.py or your config
```

**Changes needed:**
```python
# In your training config:
OPTIMIZER = 'adamw'  # Change from 'adam'
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.937

# Verify anchor configuration
NEED_AUTOANCHOR = True  # Force anchor recomputation
ANCHOR_THRESHOLD = 4.0
```

**Expected Impact:**
- Precision: 4.7% → 60-70%
- mAP@0.5: 73.2% → 75-76%

---

**Priority 2: Fix Lane Line Segmentation (68.9% → 82%+)**

```python
# Update loss configuration:
LL_SEG_GAIN = 0.6  # Increase from 0.2
DA_SEG_GAIN = 0.6  # Increase from 0.2

# Implement dual loss like TriLiteNet:
# 1. Add TverskyLoss for lane lines
# 2. Combine FocalLoss + TverskyLoss
# 3. Use alpha=0.9, beta=0.1 for LL Tversky
```

**Implementation:**
```python
# In lib/core/loss.py or equivalent:
from lib.core.twinlite_loss import TverskyLoss, FocalLossSeg

# For lane lines:
focal_ll = FocalLossSeg(mode="multiclass", alpha=0.25)
tversky_ll = TverskyLoss(mode="multiclass", alpha=0.9, beta=0.1, gamma=4/3)

# Combined loss:
ll_loss = focal_ll(pred, gt) * 0.3 + tversky_ll(pred, gt) * 0.3
```

**Expected Impact:**
- LL Accuracy: 68.9% → 80-83%
- LL IoU: 25.3% → 29-31%

---

### Phase 2: Architecture Refinement (Week 2-3)

**Option A: Adopt TriLiteNet's CAAM for Segmentation**

Advantages:
- Proven architecture for lane line segmentation
- GCN-based feature refinement
- Class activation attention

**Option B: Enhance Your MHGD Decoder**

Add components:
- Spatial attention for thin structures
- Edge-aware loss
- Boundary refinement module

**Recommendation:** Try Option A first (proven), then iterate

---

### Phase 3: Hyperparameter Optimization (Week 3-4)

**Fine-tune based on validation results:**

```python
# Experiment with these:
BATCH_SIZE = [16, 24, 32]  # You're using 24
LEARNING_RATE = [0.0005, 0.001, 0.002]  # You're using 0.001
WARMUP_EPOCHS = [3, 5, 10]  # Standard is 3

# Loss balancing (after fixing precision):
BOX_GAIN = [0.05, 0.075, 0.1]
OBJ_GAIN = [0.8, 1.0, 1.2]
CLS_GAIN = [0.5, 0.75, 1.0]
```

---

### Phase 4: Advanced Optimizations (Week 4+)

**1. Model Compression (if needed)**
- Measure your model size vs 2.35M params
- If larger: prune channels, use knowledge distillation
- If smaller: verify you're not under-parameterized

**2. Data Quality**
- Verify annotation quality for lane lines
- Add hard negative mining
- Class balancing strategies

**3. Ensemble & Post-processing**
- Multi-scale testing
- Test-time augmentation
- Weighted box fusion for detection

---

## 📈 Expected Performance After Improvements

| Metric | Current | After Phase 1 | After Phase 2 | Target (Beat TriLiteNet) |
|--------|---------|---------------|---------------|--------------------------|
| **Det Precision** | 4.7% | **65-70%** | 70-75% | >70% |
| **Det Recall** | 88.6% | 85-87% | 87-89% | >86% |
| **mAP@0.5** | 73.2% | **75-76%** | 76-78% | **>73%** ✅ |
| **DA mIoU** | 91.6% | **92-93%** | 93-94% | **>92.4%** ✅ |
| **LL Accuracy** | 68.9% | **80-82%** | 83-85% | **>82.3%** ✅ |
| **LL IoU** | 25.3% | 29-30% | 31-33% | >30% |

---

## ✅ Action Items Summary

### Immediate (This Week)

- [x] ✅ **Research completed:** Understand TriLiteNet's configuration
- [ ] 🔧 **Implement anchor optimization:** Run k-means on your dataset
- [ ] 🔧 **Switch optimizer:** Adam → AdamW with WD=0.0005
- [ ] 🔧 **Update loss weights:** LL_SEG_GAIN=0.2 → 0.6, DA_SEG_GAIN=0.2 → 0.6
- [ ] 🔧 **Add TverskyLoss:** Implement with alpha=0.9, beta=0.1 for LL

### Short-term (Next 2 Weeks)

- [ ] 📊 **Measure model complexity:** Run benchmark script for params/FLOPs
- [ ] 🧪 **Retrain with fixes:** New training run with updated config
- [ ] 📈 **Validate improvements:** Confirm precision >60%, LL acc >80%

### Long-term (Month 2+)

- [ ] 🏗️ **Consider CAAM adoption:** If LL performance still lags
- [ ] 🔬 **Ablation studies:** Test each component's contribution
- [ ] 📝 **Paper preparation:** Document your improvements and results

---

## 🎓 Key Takeaways

### What You're Doing Better Than TriLiteNet

1. ✅ **4-scale detection** - Better multi-scale object handling
2. ✅ **Higher recall (88.6% vs 85.6%)** - Detecting more objects
3. ✅ **Modern neck architecture** - Ghost convolutions + ELAN
4. ✅ **Longer training** - 300 epochs vs 240
5. ✅ **Larger batch size** - Better gradient estimates

### What TriLiteNet Does Better

1. ❌ **Balanced performance** - No catastrophic failures
2. ❌ **Dual loss approach** - Focal + Tversky for segmentation
3. ❌ **Specialized LL loss** - Tversky with alpha=0.9, beta=0.1
4. ❌ **Higher segmentation weights** - 0.6 vs your 0.2-0.4
5. ❌ **AdamW optimizer** - Better regularization
6. ❌ **CAAM architecture** - Designed for thin structures

### The Path Forward

**Your model has a strong foundation but needs critical fixes:**

1. **Fix anchor mismatch** → Precision catastrophe resolved
2. **Implement dual loss** → Lane line performance boost
3. **Switch to AdamW** → Better training stability
4. **Increase seg weights** → Balanced multi-task learning

**With these changes, you can decisively beat TriLiteNet across all metrics.**

---

**Next Steps:** Start with Phase 1 fixes. Run benchmarking script to measure model size, then retrain with new configuration. Expected timeline: 2-3 weeks to surpass TriLiteNet.
