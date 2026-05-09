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
