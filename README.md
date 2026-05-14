# SkeleGCN-CSL

**Skeleton-based GCN for Real-Time Chinese Sign Language Recognition**

A hybrid deep learning system for isolated Chinese Sign Language (CSL-500) recognition, combining Graph Convolutional Networks (GCN), Temporal Convolutional Networks (TCN), Bidirectional LSTM, and Multi-Head Self-Attention. Supports real-time inference via webcam using MediaPipe for skeleton extraction.

> 📌 **Note:** This is the initial GCN-based version of the project, developed as a computer vision course design. It serves as the baseline for ongoing SCI-level research. The research version with signer-invariant features and 3D-to-2D knowledge distillation is under active development in a separate branch.

---

## ✨ Highlights

- **Hybrid architecture** — GCN captures spatial joint topology; TCN + BiLSTM + Attention model temporal dynamics end-to-end
- **839K parameter model** — lightweight enough for RTX 3060 Laptop GPU (6 GB VRAM)
- **86.30% Top-1 / 98.21% Top-5** accuracy on CSL-500 (5-fold cross-validation, std ±0.42%)
- **Real-time capable** — ~10 ms per sample inference; 144 samples/sec throughput
- **Plug-and-play webcam** — MediaPipe keypoint extraction → model input pipeline included

---

## 🏗️ Architecture

The model `IntermediateCSL500Model` consists of four stacked modules:

```
Input: 17 joints × 2D coordinates (34-dim per frame)
        ↓
[EnhancedGCN]         — spatial topology modeling (16 skeleton edges)
        ↓
[TemporalConvBlock ×2] — local temporal pattern extraction (residual 1D conv)
        ↓
[Bidirectional LSTM]  — long-range sequence dependencies
        ↓
[Multi-Head Attention] — global context fusion (8 heads)
        ↓
[Classifier FC]       — 500-class output
```

| Module | Output Dim | Parameters |
|---|---|---|
| Input Projection | 34 | 578 |
| GCN Stack | 64 | ~31K |
| Temporal Conv | 256 | ~618K |
| BiLSTM | 128 | 82K |
| Attention | 128 | 66K |
| Classifier | 500 | ~41K |
| **Total** | — | **~839K** |

---

## 📁 Project Structure

```
SkeleGCN-CSL/
│
├── test_translation.py           # 🚀 Entry point — real-time recognition
│
├── src/
│   ├── translator.py             # Inference coordinator
│   ├── kinect_style_preprocessor.py  # Webcam capture + keypoint preprocessing
│   └── models/
│       ├── csl500_model.py       # Core GCN+LSTM model definition
│       ├── hand_model.py         # MediaPipe hand feature extraction
│       └── pose_model.py         # MediaPipe pose feature extraction
│
├── train/
│   ├── train.py                  # Training script (5-fold CV, AdamW, cosine LR)
│   ├── model_check.py            # Evaluation script
│   ├── data_preprocess.py        # Raw skeleton data preprocessing
│   ├── data_loader.py            # Dataset & DataLoader (variable-length padding)
│   └── video_to_model_input.py   # Video → model input conversion utility
│
├── model_checkpoints/
│   └── best_model_fold0.pth      # ← Place your trained weights here
│
└── data/
    └── CSL-500/
        └── processed/
            ├── csl500_data.npy   # ← Preprocessed skeleton sequences
            └── csl500_labels.npy # ← Corresponding class labels
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install torch torchvision torch-geometric
pip install mediapipe opencv-python numpy tqdm matplotlib colorlog Pillow
```

> Tested on: Python 3.9.7 · PyTorch 1.11.0+cu113 · CUDA 11.2 · Windows 10

### 2. Prepare data & model weights

Place your preprocessed data files and trained model checkpoint:

```
data/CSL-500/processed/csl500_data.npy
data/CSL-500/processed/csl500_labels.npy
model_checkpoints/best_model_fold0.pth
```

Verify paths in your config file before running.

### 3. Real-time recognition (webcam)

```bash
python test_translation.py
```

### 4. Train from scratch

```bash
cd train
python train.py
```

Training config:
- Optimizer: AdamW · LR: 1e-4 · Weight decay: 1e-4
- Scheduler: CosineAnnealingLR (min LR: 1e-6)
- Epochs: 80–100 · Batch size: 10
- Validation: 5-fold cross-validation
- ~2.5 hrs per fold on RTX 3060 Laptop

### 5. Evaluate a checkpoint

```bash
cd train
python model_check.py
```

---

## 📊 Results on CSL-500

| Metric | Score |
|---|---|
| Top-1 Accuracy | **86.30%** |
| Top-5 Accuracy | **98.21%** |
| Cross-val Std | ±0.42% |
| Inference Speed | ~144 samples/sec |
| Inference Latency | <10 ms/sample |

5-fold breakdown:

| Fold | Val Accuracy |
|---|---|
| Fold 1 | 85.92% |
| Fold 2 | 86.43% |
| Fold 3 | 86.15% |
| Fold 4 | 87.04% |
| Fold 5 | 85.96% |

---

## 🗂️ Dataset

**CSL-500 (Chinese Sign Language — 500 words)**
- 500 isolated sign classes
- 50 signers (P01–P50), 5 repetitions each
- 125,000 samples total
- Skeleton data: 25 body joints (Kinect-style) and 17-joint subset used in this version

The dataset is not included in this repository. Please refer to the [USTC CSL dataset page](http://home.ustc.edu.cn/~pjh/openresources/cslr-dataset-2015/index.html) for access.

---

## ⚠️ Known Limitations

- **Signer-dependent evaluation** — This version uses 5-fold cross-validation over all 50 signers, meaning train and test sets share signers. This inflates accuracy relative to strict signer-independent protocols. See the research branch for a corrected signer-independent split (P01–P40 train / P41–P50 test).
- **2D only** — Skeleton input uses (x, y) coordinates from MediaPipe; no depth information is used.
- **MediaPipe joint mismatch** — A mapping layer bridges MediaPipe's 33-joint output to the 17-joint format expected by the model.
- **Windows-specific** — Multiprocessing `num_workers` should be set to 0 on Windows to avoid pickle errors.

---

## 🔭 What's Next

This repository is the **v1 course-design baseline**. Active research directions building on this foundation:

- [ ] Strict signer-independent evaluation (P01–P40 / P41–P50 split)
- [ ] Signer-invariant feature design (shoulder-width normalization, joint angles, relative velocity)
- [ ] 3D-to-2D knowledge distillation from Kinect teacher to 2D student network
- [ ] ST-GCN baseline comparison
- [ ] Targeting SCI Q2 journal submission (EAAI / Neurocomputing)

---

## 📄 Citation

If you find this code useful for your research, please consider citing:

```bibtex
@misc{skelgcn-csl-2025,
  title     = {SkeleGCN-CSL: Skeleton-based GCN for Chinese Sign Language Recognition},
  author    = {[Your Name]},
  year      = {2025},
  note      = {Undergraduate course design project. GitHub repository.},
  url       = {https://github.com/[your-username]/SkeleGCN-CSL}
}
```

---

## 📜 License

This project is released for academic and educational use. See `LICENSE` for details.
