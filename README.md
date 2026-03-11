# Edge-LightAnomalyDetection

[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the complete implementation and experimental materials for reproducing the paper **"Edge-1DCNN-LSTM: A Lightweight 1DCNN-LSTM Hybrid Model for Efficient Time Series Anomaly Detection on Edge Devices"**.

The project addresses the challenge of deploying deep learning models on resource-constrained edge devices for industrial IoT time series anomaly detection, proposing:

1. **Lightweight 1DCNN-LSTM Hybrid Model**: ~12.4K parameters, F1-Score 0.9939 on simulated dataset
2. **Edge-Cloud Collaborative Inference Framework**: 81.96% energy savings with LSTM sentry model pre-screening
3. **Hybrid Precision Quantization**: INT8 quantization for CNN layers, achieving 42% inference speedup

## Key Innovations

| Innovation | Technical Approach | Experimental Result |
|------------|-------------------|---------------------|
| Hardware-aware Lightweight Architecture | Minimal 1D-CNN + Shallow LSTM | 12.4K params, F1=0.9939 |
| Edge-Cloud Collaborative Framework | LSTM sentry model pre-screening | 81.96% energy savings |
| Hybrid Precision Quantization | CNN INT8 + LSTM FP32 | 1.42x speedup, <0.1% accuracy loss |

## Project Structure

```
Edge-LightAnomalyDetection/
├── README.md                    # Project documentation
├── LICENSE                      # MIT License
│
├── cooperative_framework/       # Edge-cloud collaborative inference framework
│   ├── __init__.py
│   ├── config.py               # Configuration parameters
│   ├── cooperative_inference.py # Collaborative inference logic
│   ├── communication.py        # Communication module
│   ├── energy_monitor.py       # Energy monitoring
│   └── realistic_energy_simulation.py  # Energy simulation script
│
├── datasets/                    # Datasets
│   ├── simulate/               # Simulated dataset (50,000 samples)
│   │   ├── train_data.txt
│   │   ├── val_data.txt
│   │   ├── test_data.txt
│   │   ├── training_history.json
│   │   └── *.pth              # Data processing models
│   └── NASA/                   # NASA C-MAPSS FD001
│       ├── train_FD001.txt
│       ├── test_FD001.txt
│       └── ...
│
├── Simulate/                    # Simulated dataset experiments
│   ├── OneDCNN-LSTM/           # Main hybrid model
│   ├── OneDCNN-LSTM_Quantizated/ # Quantization experiments
│   ├── Only_LSTM/              # Ablation: Pure LSTM
│   ├── Only_OneDCNN/           # Ablation: Pure 1D-CNN
│   ├── IsolationForest/        # Baseline: Isolation Forest
│   └── Rule-Based/             # Baseline: Rule-based method
│
├── FD001/                       # NASA dataset experiments
│   └── (same structure as Simulate/)
│
├── tests/                       # Testing and validation scripts
│   ├── test_cooperative_framework.py
│   ├── run_full_validation.py
│   ├── run_validation_with_real_models.py
│   └── conv_kernel_experiment.py
│
├── deployment/                   # Edge deployment
│   └── raspberry_pi/
│
├── results/                      # Experimental results
│   ├── paper_figures/           # Paper figures (Chinese)
│   ├── english_figures/         # Paper figures (English)
│   ├── final_optimization/      # Collaborative framework optimization
│   ├── cooperative_energy_simulation/ # Energy simulation results
│   ├── cooperative_validation/  # Collaborative validation
│   ├── performance_data.csv     # Performance data
│   ├── quantization_report.md   # Quantization report
│   └── performance_comparison_fixed.md
│
└── scripts/                      # Utility scripts
    ├── convert_FD001_train.py   # NASA data preprocessing
    ├── convert_FD001_test.py    # NASA test data conversion
    ├── create_dirs.py           # Directory creation tool
    ├── cleanup_temp.py          # Temp file cleanup
    ├── paper_figures_generator.py # Paper figure generation
    └── generate_english_figures.py # English figure generation

```

## Quick Start

### Requirements

- Python 3.9+
- PyTorch 2.0.0+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn

### Installation

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Training

```bash
# Train main model
cd Simulate/OneDCNN-LSTM
python ann_train.py

# Evaluate model
python ann_evaluate.py

# Model quantization
cd ../OneDCNN-LSTM_Quantizated
python ann_quantization.py
```

### Run Collaborative Framework Simulation

```bash
cd cooperative_framework
python realistic_energy_simulation.py
```

## Experimental Results

### Model Performance Comparison (Simulated Dataset)

| Model | F1-Score | AUC-ROC | Parameters | Inference Latency |
|-------|----------|---------|------------|-------------------|
| **Edge-1DCNN-LSTM** | **0.9939** | **0.9992** | 12,385 | 59.28ms |
| LSTM-only | 0.9912 | 0.9987 | 8,481 | 21.82ms |
| 1D-CNN-only | 0.9020 | 0.9601 | 2,625 | 25.41ms |
| Isolation Forest | 0.9027 | 0.9498 | - | 15ms |
| Rule-based | 0.8542 | 0.9255 | - | <1ms |

### Quantization Effects

| Metric | FP32 | INT8 | Change |
|--------|------|------|--------|
| Model Size | 192KB | 65.1KB | -66.1% |
| Inference Latency | 59.28ms | 41.76ms | -29.5% |
| F1-Score | 0.9939 | 0.9935 | -0.04% |

### Edge-Cloud Collaborative Energy Efficiency

| Metric | Value |
|--------|-------|
| Wake-up Rate | 8.55% |
| Energy Savings | 81.96% |
| System F1-Score | 0.9944 |

## Dataset Description

### Simulated Dataset

- Total samples: 50,000
- Anomaly ratio: 15%
- Anomaly types:
  - Gradual Drift (26.7%)
  - Sudden Spike (24.0%)
  - Persistent Shift (25.3%)
  - Periodic Disruption (24.0%)

### NASA C-MAPSS FD001

- Training samples: 20,000
- Test anomaly ratio: 2.54%
- Source: NASA turbofan engine degradation benchmark

## Model Architecture

```
Input (Batch × 10 × 1)
    ↓
1D Convolution (16 filters, kernel=3, padding=1)
    ↓
Batch Normalization + ReLU
    ↓
Max Pooling (kernel=2, stride=2)
    ↓
Reshape (Batch × 5 × 16)
    ↓
2-layer LSTM (hidden_size=32, dropout=0.2)
    ↓
Dropout (p=0.2)
    ↓
Fully Connected Layer
    ↓
Sigmoid Activation
    ↓
Output (Batch × 1)
```

**Parameters**:
- Total Parameters: ~12,385
- CNN Channels: 16
- LSTM Hidden Units: 32
- LSTM Layers: 2
- Dropout Rate: 0.2
- Sequence Length: 10

## Citation

```bibtex
@article{edge1dcnnlstm2026,
  title={Edge-1DCNN-LSTM: A Lightweight 1DCNN-LSTM Hybrid Model for Efficient Time Series Anomaly Detection on Edge Devices},
  author={PandaKing},
  year={2026}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- NASA for providing the C-MAPSS dataset
- PyTorch deep learning framework
