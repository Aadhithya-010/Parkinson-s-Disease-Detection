# Study Guide: Parkinson's Disease Detection Using Deep Learning

Use this guide to study every technical decision in the project and answer interview or submission viva questions clearly.

---

## 1. Data Ingestion, Auditing, and Leakage (`src/data/audit.py`)

### Why deduplicate with MD5 *before* splitting?
If identical images (or duplicate slices) exist in the raw download and you split first, one copy can land in `train` and the other in `test`. That causes **data leakage**, inflating test accuracy without real generalization. Hashing raw file bytes with MD5 removes exact duplicates before any split happens.

### Why check for `subject_id` (participant-level splitting)?
In medical datasets, one patient often has multiple scans or 2D slices. If slice 10 is in `train` and slice 11 of the **same patient** is in `test`, the model can memorize patient-specific anatomy rather than Parkinson's markers. When `PatientID` metadata is available, `GroupShuffleSplit` keeps all images from a given subject inside a single split.

### Why keep `validate/` instead of training directly on `test/`?
- `train`: updates model weights via backpropagation.
- `validate`: guides `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` without touching test data.
- `test`: evaluated **once** at the end (`shuffle=False`) for an unbiased performance comparison.

---

## 2. Exploratory Data Analysis (`src/data/eda.py`)

### What is the "pixel intensity shortcut" check?
If all Parkinson's images came from one scanner/source with darker backgrounds and all healthy images came from another with brighter backgrounds, a CNN can achieve high accuracy simply by measuring average brightness instead of disease features. Plotting training intensity histograms per class checks for this bias early.

### Why check class balance and compute `class_weight`?
With imbalanced classes, a model can minimize binary cross-entropy by always predicting the majority class (high accuracy, **0% recall** on the minority class). Computing balanced weights (`n_samples / (n_classes * count_c)`) scales the loss penalty inversely with class frequency.

---

## 3. OpenCV Preprocessing & `tf.data` (`src/data/pipeline.py`)

### Why preserve aspect ratio instead of naive stretching?
Direct `resize((224, 224))` distorts shapes (turning circular spirals or brain structures into ovals). Scaling the longer side to 224 (`cv2.resize`) and padding (`cv2.copyMakeBorder`) preserves geometric proportions. Padding with the median border pixel color avoids introducing harsh white frames around dark MRI scans or black frames around light drawings.

### Why are `preprocess_input` functions different for CNN, VGG16, and ResNet50?
- **CustomCNN:** expects inputs scaled to `[0, 1]` (`x / 255.0`).
- **VGG16 / ResNet50:** pretrained on ImageNet using Caffe-style channel ordering (`RGB -> BGR`) and dataset mean subtraction (`[103.939, 116.779, 123.68]`). Passing `[0, 1]` images into ImageNet `VGG16`/`ResNet50` silently degrades pretrained feature extraction.

### Why augment *only* the training set?
Augmentation (horizontal flip, small rotation/zoom/translation/contrast) increases effective training diversity. Validation and test sets must remain deterministic and unmodified so evaluation reflects real inputs.

---

## 4. Custom CNN Architecture (`src/models/cnn.py`)

### Layer-by-layer breakdown
1. **`Conv2D(32/64/128, 3x3)`**: learns spatial edge, texture, and structural filters.
2. **`BatchNormalization`**: normalizes activations per mini-batch during training and tracks moving mean/variance for inference.
3. **`ReLU`**: introduces non-linearity (`max(0, x)`).
4. **`MaxPooling2D(2x2)`**: halves spatial resolution `(224 -> 112 -> 56 -> 28)`, increasing receptive field and reducing compute.
5. **`GlobalAveragePooling2D` vs `Flatten`**:
   - `Flatten()` on `(28, 28, 128)` produces `100,352` features → millions of Dense parameters → severe overfitting on small datasets.
   - `GlobalAveragePooling2D()` averages each of the 128 feature maps to 1 number → `128` features → compact parameter count.
6. **`Dropout(0.5)`**: randomly zeroes 50% of activations during training to reduce co-adaptation.
7. **`Dense(1, activation="sigmoid")`**: outputs $P(\text{parkinson} \mid x) \in (0, 1)$, trained with `binary_crossentropy`.

---

## 5. Transfer Learning: VGG16 & ResNet50 (`src/models/transfer.py`)

### Why train in two phases?
- **Phase 1 (Frozen backbone, LR = `1e-3`):** The new `Dense` classification head starts with random weights. If the backbone were unfrozen immediately, large random gradients from the untrained head would corrupt pretrained ImageNet filters.
- **Phase 2 (Fine-tuning top layers, LR = `1e-5`):** Once the head is aligned, we unfreeze the top 20 layers with a 100x smaller learning rate so high-level feature maps adapt gently to medical/handwriting patterns.

### Why keep `BatchNormalization` layers frozen during ResNet50 fine-tuning?
With small batch sizes (e.g. `16`) and small datasets, updating `BatchNormalization` moving statistics overwrites the stable ImageNet statistics with noisy mini-batch estimates, often causing validation loss spikes or collapsed predictions. Keeping `bn.trainable = False` locks BN layers in inference mode.

---

## 6. Evaluation Metrics (`src/evaluation/metrics.py`)

Given confusion matrix `[[TN, FP], [FN, TP]]` (where positive class `1` = `parkinson`):
- **Accuracy** $= \frac{TP + TN}{TP + TN + FP + FN}$ — misleading when classes are imbalanced or a model predicts all samples as one class.
- **Precision** $= \frac{TP}{TP + FP}$ — of predicted Parkinson's cases, how many are truly Parkinson's.
- **Recall (Sensitivity)** $= \frac{TP}{TP + FN}$ — of true Parkinson's cases, how many the model detects. Critical in screening because missed cases (`FN`) are costly.
- **Specificity** $= \frac{TN}{TN + FP}$ — of true healthy cases, how many are correctly cleared.
- **F1-score** $= 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ — harmonic mean balancing precision and recall.
- **ROC-AUC** — threshold-independent ranking quality across all probability cutoffs (`0.5` is random guessing).

---

## 7. Grad-CAM Explainability (`src/models/gradcam.py`)

### How Grad-CAM works
1. Pass an image forward up to the last convolutional layer $A^k$ (shape $H \times W \times K$) and final pre-threshold output $y$ (`P(parkinson)`).
2. Compute gradients $\frac{\partial y}{\partial A^k}$ via `tf.GradientTape`.
3. Global-average-pool the gradients spatially to get importance weights $\alpha_k = \frac{1}{HW} \sum_i \sum_j \frac{\partial y}{\partial A^k_{ij}}$.
4. Compute weighted feature combination and apply ReLU:
   $$L_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k A^k\right)$$
5. Normalize to `[0, 1]`, resize to `224x224` with OpenCV, apply `COLORMAP_JET`, and blend over the input image (`cv2.addWeighted`).

### How to interpret Grad-CAM critically
If heatmaps focus on image borders, text watermarks, or empty corners instead of anatomical structures (or handwriting strokes), the model is relying on dataset artifacts rather than meaningful pathology.
