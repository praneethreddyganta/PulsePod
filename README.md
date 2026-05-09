# 🫀 PulsePod — Real-Time ECG Beat Classification & Simulation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8--3.12-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=flat-square&logo=streamlit" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Research%20%2F%20Demo-yellow?style=flat-square" />
</p>

---

## 📌 Project Description

**PulsePod** is an end-to-end research and demonstration platform for beat-level ECG classification (Normal vs. Abnormal) built on top of the MIT-BIH Arrhythmia Database and the PTB-XL dataset. It integrates a complete preprocessing pipeline, cross-dataset stratified augmentation, multiple deep learning model architectures (1D-CNN, CNN+LSTM, and an experimental Spiking Neural Network), and a real-time Streamlit simulation dashboard for live inference and visualization.

This repository is designed for researchers, biomedical ML engineers, and students who want a reproducible, well-structured baseline for ECG arrhythmia detection — from raw waveform to deployed inference.

---

## 🧩 Problem Statement

Cardiac arrhythmias are among the leading causes of sudden cardiac death worldwide. Automated, real-time ECG beat classification can enable earlier detection of abnormal rhythms without requiring expert cardiologist intervention. However, building robust classifiers is challenged by:

- High class imbalance (abnormal beats are rare)
- Domain shift across different ECG acquisition hardware and patient populations
- The need for low-latency, lightweight inference suitable for wearable and edge devices

PulsePod addresses these challenges through cross-dataset training, focal-loss experiments, SNN-based inference, and a stratified augmentation pipeline.

---

## ✨ Features

- **Real-time ECG beat simulation** with live classification via a Streamlit dashboard
- **Multiple model architectures**: 1D-CNN, CNN+LSTM (temporal context), and experimental Spiking Neural Network (SNN)
- **Cross-dataset evaluation**: trained and tested across MIT-BIH and PTB-XL
- **Pretrained PyTorch checkpoints** ready for inference out of the box
- **Data augmentation pipeline**: Gaussian noise, time-shifting, amplitude scaling for minority-class oversampling
- **HRV feature extraction** integrated into the preprocessing pipeline
- **Focal loss experiments** for handling class imbalance
- **Confusion matrix and waveform visualization** within the live demo

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8 – 3.12 |
| Deep Learning | PyTorch |
| SNN | snnTorch |
| Web Demo | Streamlit |
| ECG Processing | wfdb, neurokit2 |
| Numerical | NumPy, SciPy, pandas |
| ML Utilities | scikit-learn |
| Visualization | Matplotlib, seaborn |

---

## 📁 Complete Project Structure

```
PulsePod/
│
├── app.py                          # Main Streamlit demo, model classes, inference loop
├── original_app.py                 # Reference/earlier Streamlit configuration
├── cnn_requirements.txt            # Core dependencies (CNN demo + training)
├── snn_requirements.txt            # Additional SNN dependencies (snnTorch)
│
├── data/
│   ├── create_unified_test_set.py  # Script to merge MIT-BIH + PTB-XL test arrays
│   ├── mit_bih/
│   │   └── X_test.npy              # Preprocessed MIT-BIH beat arrays
│   ├── ptb_xl/
│   │   └── X_test.npy              # Preprocessed PTB-XL beat arrays
│   └── unified_test_set/
│       ├── X_test_unified.npy      # Merged test features
│       └── y_test_unified.npy      # Merged test labels
│
├── models/
│   ├── final_robust_cnn.pt             # Pretrained 1D-CNN checkpoint
│   ├── final_robust_cnn_focal_loss.pt  # 1D-CNN trained with focal loss
│   ├── final_robust_cnn_lstm.pt        # Pretrained CNN+LSTM checkpoint
│   └── final_robust_snn.pt             # Pretrained SNN checkpoint (experimental)
│
├── research/
│   ├── 01_exploratory_data_analysis.ipynb       # EDA, signal plotting, class distribution
│   ├── 02_dataset_creation_pipeline.ipynb       # Beat segmentation, HRV extraction
│   ├── 03_data_augmentation_stratification.py   # Stratified splits + augmentation
│   ├── 04_model_training_experiments.ipynb      # Training loops, metrics, loss curves
│   └── README.md                                # Research notebook index
│
└── logs/
    └── streamlit_events.log        # Runtime event log from the Streamlit demo
```

---

## 🏗️ System Workflow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAW ECG SIGNALS                              │
│              (MIT-BIH Arrhythmia DB + PTB-XL DB)                   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                           │
│                                                                     │
│  ┌──────────────┐   ┌────────────────┐   ┌──────────────────────┐  │
│  │  R-Peak      │──▶│ Beat           │──▶│ Resample to          │  │
│  │  Detection   │   │ Segmentation   │   │ 187 Samples          │  │
│  └──────────────┘   └────────────────┘   └──────────────────────┘  │
│                                                    │                │
│                                                    ▼                │
│                                    ┌───────────────────────────┐   │
│                                    │  HRV Feature Extraction   │   │
│                                    └───────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DATASET UNIFICATION                             │
│                                                                     │
│       data/mit_bih/X_test.npy  +  data/ptb_xl/X_test.npy          │
│                         │                                           │
│                         ▼                                           │
│              create_unified_test_set.py                             │
│                         │                                           │
│                         ▼                                           │
│          data/unified_test_set/X_test_unified.npy                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  DATA AUGMENTATION & SPLIT                          │
│                                                                     │
│   Stratified Train/Val/Test Split (scikit-learn)                    │
│   Minority Class (Abnormal) Oversampling:                           │
│     • Gaussian Noise Injection                                      │
│     • Time-Shift Augmentation                                       │
│     • Amplitude Scaling                                             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING                                 │
│                                                                     │
│    ┌────────────┐   ┌─────────────────┐   ┌────────────────────┐   │
│    │  ECG_CNN   │   │  CNN_LSTM_Model  │   │    SNN_Model       │   │
│    │  (1D-CNN)  │   │  (Seq. Context)  │   │  (Experimental)    │   │
│    └────────────┘   └─────────────────┘   └────────────────────┘   │
│                                                                     │
│    Optional: Focal Loss for class imbalance                         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   STREAMLIT DEMO (app.py)                           │
│                                                                     │
│   load_cached_model() ──────────▶ models/*.pt checkpoint            │
│   load_cached_test_data() ──────▶ unified_test_set/*.npy            │
│                                                                     │
│   Per-Beat Loop:                                                    │
│   Sample Beat ──▶ Inference ──▶ create_beat_plot()                  │
│                              ──▶ handle_alert()                     │
│                              ──▶ create_confusion_matrix_plot()      │
│                              ──▶ streamlit_events.log               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Model Architectures

### 1. ECG_CNN — 1D Convolutional Neural Network

```
Input Beat: [Batch, 1, 187]
     │
     ▼
┌─────────────────────────────────────────────┐
│  Conv1D(1 → 32, kernel=5)  + BatchNorm + ReLU│
│  MaxPool1D(2)                                │
├─────────────────────────────────────────────┤
│  Conv1D(32 → 64, kernel=5) + BatchNorm + ReLU│
│  MaxPool1D(2)                                │
├─────────────────────────────────────────────┤
│  Conv1D(64 → 128, kernel=3) + BatchNorm + ReLU│
│  MaxPool1D(2)                                │
├─────────────────────────────────────────────┤
│  AdaptiveAvgPool1D(1)                        │
│  Flatten → [Batch, 128]                      │
├─────────────────────────────────────────────┤
│  FC(128 → 64) + ReLU + Dropout(0.5)         │
│  FC(64 → 2)                                  │
└─────────────────────────────────────────────┘
     │
     ▼
Output: [Normal | Abnormal]
```

### 2. CNNFeatureExtractor + CNN_LSTM_Model — Temporal Sequence Classifier

```
Input Sequence: [Batch, SeqLen, 187]  (SeqLen consecutive beats)
     │
     ▼
┌────────────────────────────────────────────────┐
│           CNNFeatureExtractor (per beat)        │
│  Conv1D blocks → AdaptiveAvgPool → Embedding   │
│  Output: [Batch, SeqLen, EmbedDim]              │
└────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────┐
│           Bidirectional LSTM                   │
│  Input:  [Batch, SeqLen, EmbedDim]             │
│  Hidden: 128 units, 2 layers, Dropout          │
│  Output: [Batch, SeqLen, 256]                  │
│  Take last timestep → [Batch, 256]             │
└────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────┐
│  FC(256 → 128) + ReLU + Dropout(0.5)           │
│  FC(128 → 2)                                   │
└────────────────────────────────────────────────┘
     │
     ▼
Output: [Normal | Abnormal]
```

The CNN+LSTM model captures short-term temporal dependencies across consecutive beats, making it sensitive to rhythm-level abnormalities rather than purely morphological features in a single beat.

### 3. SNN_Model — Experimental Spiking Neural Network

```
Input Beat: [Batch, 1, 187]  (presented over T timesteps)
     │
     ▼
┌──────────────────────────────────────────────────┐
│  Encode: Rate-coded spike trains over T steps     │
│          Poisson encoding or latency encoding     │
└──────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────┐
│  Spiking Conv Layer 1 (snntorch LIF neurons)     │
│  Conv1D(1 → 32) + Leaky Integrate-and-Fire      │
├──────────────────────────────────────────────────┤
│  Spiking Conv Layer 2                            │
│  Conv1D(32 → 64) + LIF                          │
├──────────────────────────────────────────────────┤
│  Flatten → Spiking FC Layer                      │
│  FC(64*N → 128) + LIF                           │
├──────────────────────────────────────────────────┤
│  Readout Layer: FC(128 → 2)                      │
│  Accumulate membrane potential over T timesteps  │
└──────────────────────────────────────────────────┘
     │
     ▼
Output: Spike count / membrane potential → [Normal | Abnormal]
```

> ⚠️ **Experimental**: Requires `snnTorch`. Install with `snn_requirements.txt`. Performance may vary; see `research/04_model_training_experiments.ipynb` for benchmarks.

---

## 🔬 Data Processing Pipeline

### Step 1 — Beat Segmentation (`research/02_dataset_creation_pipeline.ipynb`)

- Load raw ECG records from MIT-BIH and PTB-XL using `wfdb`
- Detect R-peaks using NeuroKit2's `ecg_peaks()` (Pan-Tompkins algorithm)
- Segment each beat centered on the R-peak with a fixed window
- Resample each beat to exactly **187 samples** using SciPy for uniform CNN input
- Extract RR-interval and HRV-derived features (SDNN, RMSSD, pNN50)

### Step 2 — Dataset Unification (`data/create_unified_test_set.py`)

```bash
python data/create_unified_test_set.py
```

Stacks preprocessed arrays from both datasets:
- `data/mit_bih/X_test.npy` + `data/ptb_xl/X_test.npy` → `data/unified_test_set/X_test_unified.npy`
- `data/mit_bih/y_test.npy` + `data/ptb_xl/y_test.npy` → `data/unified_test_set/y_test_unified.npy`

### Step 3 — Augmentation & Stratified Splitting (`research/03_data_augmentation_stratification.py`)

Applied to the minority class (Abnormal beats):

| Technique | Description |
|---|---|
| Gaussian Noise | Adds zero-mean Gaussian noise (σ scaled to signal amplitude) |
| Time Shift | Randomly shifts beats left/right within the 187-sample window |
| Amplitude Scaling | Scales the beat amplitude by a random factor in [0.9, 1.1] |

Stratified train/val/test split preserves class ratios across subsets.

---

## 📊 Dataset Details

| Dataset | Description | Source |
|---|---|---|
| **MIT-BIH Arrhythmia Database** | 48 two-lead Holter ECG recordings, 109,000+ annotated beats, sampled at 360 Hz | PhysioNet |
| **PTB-XL** | 21,799 12-lead clinical ECG records, 10 seconds each, sampled at 500 Hz | PhysioNet |
| **Unified Test Set** | Combined held-out test split from both datasets | Local — `data/unified_test_set/` |

**Label Mapping**: Binary classification — `0 = Normal`, `1 = Abnormal`

**Beat representation**: Each beat is stored as a 1D array of 187 float values (single-lead normalized amplitude), shaped `[N, 187]` in `.npy` format.

> Large `.npy` and `.pt` files are tracked via **Git LFS** (see `.gitattributes`).

---

## 🏋️ Training Pipeline

Training is managed through `research/04_model_training_experiments.ipynb`. Key details:

- **Loss functions**: Cross-entropy (baseline) and focal loss (`final_robust_cnn_focal_loss.pt`) for class imbalance
- **Optimizer**: Adam with learning rate scheduling
- **Evaluation metrics**: Accuracy, F1-score (macro + per-class), AUC-ROC, confusion matrix
- **Cross-dataset generalization**: Models evaluated on both MIT-BIH and PTB-XL held-out test sets independently and on the unified test set
- **Checkpoints**: Saved to `models/` after best validation F1

To reproduce training:

```bash
# Launch Jupyter and open the training notebook
jupyter notebook research/04_model_training_experiments.ipynb
```

---

## ⚡ Inference Pipeline

The Streamlit app (`app.py`) implements a real-time beat-by-beat simulation loop:

```
┌──────────────────────────────────────────────────────────┐
│                  app.py — Inference Loop                 │
│                                                          │
│  1. load_cached_model(checkpoint_path)                   │
│     └──▶ Loads ECG_CNN / CNN_LSTM_Model / SNN_Model      │
│                                                          │
│  2. load_cached_test_data(data_path)                     │
│     └──▶ Loads unified_test_set/*.npy into memory        │
│                                                          │
│  3. Per-beat loop (user-configured speed):               │
│     ├──▶ Sample next beat from test set                  │
│     ├──▶ model.eval() → forward pass → logits → label    │
│     ├──▶ create_beat_plot(beat, prediction, true_label)  │
│     ├──▶ handle_alert(prediction, threshold)             │
│     └──▶ Log to logs/streamlit_events.log                │
│                                                          │
│  4. Aggregate metrics update:                            │
│     └──▶ create_confusion_matrix_plot(predictions, labels)│
└──────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8–3.12
- Git with [Git LFS](https://git-lfs.github.com/) (required for `.npy` and `.pt` files)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/pulsepod.git
cd pulsepod

# 2. Pull large files via Git LFS
git lfs pull

# 3. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 4. Install core dependencies
pip install -r cnn_requirements.txt

# 5. (Optional) Install SNN dependencies
pip install -r snn_requirements.txt
```

---

## 📖 Usage

### 1. Prepare Processed Data

If preprocessed `.npy` files are not present, generate them:

```bash
# Run the dataset creation notebook
jupyter notebook research/02_dataset_creation_pipeline.ipynb

# Then unify the test sets
python data/create_unified_test_set.py
```

### 2. Launch the Real-Time Demo

```bash
streamlit run app.py
```

In the Streamlit sidebar:
- Select a model checkpoint from `models/`
- Configure simulation speed (beats per second)
- Choose dataset source (MIT-BIH, PTB-XL, or Unified)

### 3. Reproduce Training Experiments

```bash
jupyter notebook research/04_model_training_experiments.ipynb
```

Follow the notebook cells to train and evaluate 1D-CNN, CNN+LSTM, and SNN models.

### 4. Run Data Augmentation & Stratification Separately

```bash
python research/03_data_augmentation_stratification.py
```

---

## 📈 Results and Outputs

### Streamlit Dashboard Outputs

| Output | Function | Description |
|---|---|---|
| Beat Waveform | `create_beat_plot()` | Live plot of the current ECG beat with predicted and true label overlay |
| Confusion Matrix | `create_confusion_matrix_plot()` | Cumulative confusion matrix updated per inference step |
| Alerts | `handle_alert()` | Triggered when consecutive abnormal beats exceed a threshold |
| Event Log | `logs/streamlit_events.log` | Timestamped record of predictions, alerts, and session events |

### Model Checkpoints

| Checkpoint | Architecture | Notes |
|---|---|---|
| `final_robust_cnn.pt` | 1D-CNN (`ECG_CNN`) | Standard cross-entropy training |
| `final_robust_cnn_focal_loss.pt` | 1D-CNN (`ECG_CNN`) | Focal loss for class imbalance |
| `final_robust_cnn_lstm.pt` | CNN+LSTM (`CNN_LSTM_Model`) | Temporal sequence context |
| `final_robust_snn.pt` | SNN (`SNN_Model`) | Experimental; requires snnTorch |

> Detailed training curves, per-class F1, and AUC-ROC values are recorded in `research/04_model_training_experiments.ipynb`.

---

## 🔮 Future Improvements

- [ ] Add unit tests, CI/CD pipeline (GitHub Actions), and reproducible Docker environments
- [ ] Export models to TorchScript / ONNX for edge and wearable deployment
- [ ] Add calibrated uncertainty estimation (Monte Carlo Dropout, temperature scaling)
- [ ] Improve label harmonization between MIT-BIH AAMI categories and PTB-XL SCP codes
- [ ] Extend to multi-class arrhythmia classification (AFib, PVC, LBBB, RBBB, etc.)
- [ ] Provide prebuilt Docker images bundling snnTorch for SNN experiments
- [ ] Integrate streaming ECG support (real-time hardware feed via serial / BLE)

---

## 👥 Authors / Contributors

| Name | Contributions |
|---|---|
| **Ganta Praneeth Reddy** | ECG data preprocessing pipeline, R-peak based beat segmentation, HRV feature extraction, CNN / CNN+LSTM / SNN experimentation |
| **T. Vishwanath** | Core model development and training, focal loss experiments, model evaluation and checkpointing |
| **S. Smith** | Additional project implementation, Streamlit demo integration, and testing support |

---

## 📜 License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

## 🙏 Acknowledgements

- [PhysioNet / MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) — Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol. 2001.
- [PhysioNet / PTB-XL ECG Dataset](https://physionet.org/content/ptb-xl/1.0.3/) — Wagner et al., 2020.
- [wfdb](https://github.com/MIT-LCP/wfdb-python) — Python library for reading PhysioNet waveform data.
- [NeuroKit2](https://github.com/neuropsychology/NeuroKit) — ECG signal processing and peak detection.
- [snnTorch](https://github.com/jeshraghian/snntorch) — Deep learning with spiking neural networks in PyTorch.
- [Streamlit](https://streamlit.io/) — Interactive web application framework for ML demos.# 🫀 PulsePod — Real-Time ECG Beat Classification & Simulation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8--3.12-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=flat-square&logo=streamlit" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Research%20%2F%20Demo-yellow?style=flat-square" />
</p>

---
