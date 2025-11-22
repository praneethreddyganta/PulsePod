<div align="center">

# 💓 PulsePod
### Intelligent Edge-Ready ECG Analysis System
**Deep Learning • Neuromorphic Computing • Real-Time Deployment**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SNNTorch](https://img.shields.io/badge/SNN-Neuromorphic-purple)](https://snntorch.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

<p align="center">
  <a href="#-project-overview">Overview</a> •
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-performance--results">Results</a>
</p>
</div>

---

## 🩺 Project Overview
**PulsePod** is an end-to-end AI system designed to detect cardiac abnormalities from ECG signals with high reliability and efficiency. Unlike traditional models that overfit to specific datasets, PulsePod is built for **generalization** and **edge deployment**.

It tackles the critical challenge of **Domain Shift** by training on a massive, unified dataset combining **MIT-BIH** and **PTB-XL** records. The system features a suite of advanced models ranging from robust **1D-CNNs** to ultra-low-power **Spiking Neural Networks (SNNs)**.

---

## 🌟 Features
* **🧠 Multi-Paradigm Modeling:** Includes **1D-CNNs**, **CNN-LSTM Hybrids**, and **Spiking Neural Networks (SNNs)**.
* **🛡️ Robust Generalization:** Trained on a unified dataset to solve domain shift between hospitals.
* **⚡ Neuromorphic Intelligence:** Validated SNNs proving viability for ultra-low-power hardware.
* **🚀 Streamlit Dashboard:** Interactive web interface for visualizing ECG signals and model predictions.
* **📊 Unified Data Pipeline:** Scripts to merge and standardize diverse ECG sources.

---

## 🏗 Architecture

The system follows a modular pipeline from raw signal ingestion to edge deployment.

```mermaid
graph LR
    A[Raw ECG Data] --> B(Preprocessing & Segmentation)
    B --> C{Unified Dataset Creation}
    C --> D[Training Loop]
    D --> E[Model Zoo]
    E --> F[Deployment Dashboard]

    subgraph Models
        E1[Standalone CNN]
        E2[SNN Neuromorphic]
        E3[CNN+LSTM]
    end

    E --> E1
    E --> E2
    E --> E3
⚙️ InstallationTo ensure environment stability, requirements are split into two categories.1. Clone the RepositoryBashgit clone [https://github.com/yourusername/PulsePod.git](https://github.com/yourusername/PulsePod.git)
cd PulsePod
git lfs install
git lfs pull
2. Set up the EnvironmentOption A: For Standard CNN Models (Recommended)Use this for the main Dashboard and standard Deep Learning models.Bash# Create virtual environment
python -m venv venv_cnn
source venv_cnn/bin/activate  # or venv_cnn\Scripts\activate on Windows

# Install dependencies
pip install -r cnn_requirements.txt
Option B: For Neuromorphic SNN ExperimentsUse this specifically for running Spiking Neural Networks.Bash# Create virtual environment
python -m venv venv_snn
source venv_snn/bin/activate  # or venv_snn\Scripts\activate on Windows

# Install dependencies
pip install -r snn_requirements.txt
🚀 Usage1. Launch the DashboardVisualize the inference stream in a web interface.Bash# Ensure venv_cnn is active
streamlit run app.py
2. Data Preprocessing (Optional)The test data is already provided in .npy format in the data/ folder. If you wish to regenerate the unified test set from raw sources:Bashpython data/create_unified_test_set.py
📊 Performance & ResultsWe evaluated our models on a held-out, multi-domain test set (Unified MIT-BIH + PTB-XL) to ensure true robustness.Model ArchitecturePrecision (Abnormal)Recall (Abnormal)F1-ScoreUse CaseStandalone CNN89%86%88%Best Overall BalanceSpiking NN (SNN)85%87%86%Ultra-Low Power / High RecallCNN + LSTM91%82%87%Precision / False Alarm Reduction📂 Project StructurePlaintextPulsePod/
├── data/                           # Processed .npy datasets & Scripts
│   ├── mit_bih/
│   ├── ptb_xl/
│   ├── unified_test_set/
│   └── create_unified_test_set.py  # Script to merge datasets
├── models/                         # Pre-trained PyTorch models
│   ├── final_robust_cnn.pt
│   ├── final_robust_cnn_focal_loss.pt
│   ├── final_robust_cnn_lstm.pt
│   └── final_robust_snn.pt
├── app.py                          # Main Streamlit Dashboard
├── cnn_requirements.txt            # Standard dependencies
├── snn_requirements.txt            # SNN dependencies
└── readme.md
🔮 Future ScopeHardware Porting: Deploy SNN models to Intel Loihi or Neuromorphic chips.Multi-Class: Extend detection to specific arrhythmias (AFib, PVC, LBBB).Federated Learning: Implement privacy-preserving patient data training.<div align="center">Built with 💙 by [Vishwanath.T]</div>