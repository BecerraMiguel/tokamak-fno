# Tokamak Disruption Prediction with Neural Operators

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Week%201%20Complete-brightgreen.svg)

A machine learning system for predicting plasma disruptions in tokamak fusion reactors using Fourier Neural Operators (FNO). This project aims to develop real-time prediction capabilities that could help protect fusion devices like ITER from catastrophic disruption events.

## Project Goals

| Metric | Target | Current (Synthetic) |
|--------|--------|---------------------|
| True Positive Rate | > 90% | 100% |
| False Positive Rate | < 10% | 0% |
| Warning Time | > 20ms | ✓ |
| Inference Latency | < 10ms | ✓ |

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/BecerraMiguel/tokamak-fno.git
cd tokamak-fno
```

### 2. Create conda environment

```bash
conda create -n tokamak_fno python=3.10 -y
conda activate tokamak_fno
pip install -r requirements.txt
```

### 3. Generate synthetic data

```bash
python -c "from src.data.synthetic import SyntheticTokamakGenerator; g = SyntheticTokamakGenerator(); g.generate_dataset('data/tokamak_synthetic.h5', n_disruptive=500, n_normal=500)"
```

### 4. Train baseline model

```bash
python -c "
from src.data.loader import get_dataloaders
from src.models.baseline import BaselineCNN
from src.training.train import train_model
import torch

train_loader, val_loader = get_dataloaders('data/tokamak_synthetic.h5', batch_size=32)
model = BaselineCNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trained_model, history = train_model(model, train_loader, val_loader, device, epochs=30)
"
```

### 5. Evaluate

See `notebooks/03_baseline_evaluation.ipynb` for detailed evaluation with metrics and visualizations.

## Project Structure

```
tokamak-fno/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               
│
├── configs/                  # Configuration files (Week 2+)
│
├── data/                     # Dataset storage
│   └── tokamak_synthetic.h5  # Generated synthetic data
│
├── docs/                     # Documentation
│   └── ARCHITECTURE.md       # Technical architecture details
│
├── notebooks/                # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_training.ipynb
│   └── 03_baseline_evaluation.ipynb
│
├── results/                  # Training outputs and visualizations
│   ├── best_model.pt         # Trained model checkpoint
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── baseline_metrics.csv
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── synthetic.py      # Synthetic data generator
│   │   └── loader.py         # Dataset and DataLoader
│   ├── models/
│   │   ├── __init__.py
│   │   └── baseline.py       # BaselineCNN model
│   └── training/
│       ├── __init__.py
│       ├── train.py          # Training loop
│       └── evaluate.py       # Evaluation metrics
│
└── tests/                    # Unit tests (Week 4)
```

## 🔬 Technical Overview

### The Problem: Plasma Disruptions

Plasma disruptions are sudden losses of confinement in tokamak fusion reactors that can:
- Release megajoules of energy in milliseconds
- Generate electromagnetic forces that damage reactor components
- Create runaway electron beams that can penetrate walls
- Cost millions of dollars in repairs and downtime

For ITER to be economically viable, disruption rates must be kept below 1%.

### Our Approach

We use deep learning to predict disruptions from plasma diagnostic signals before they occur, providing sufficient warning time (>20ms) to activate mitigation systems.

**Diagnostic Signals Used:**
- `ip` - Plasma current [MA]
- `betan` - Normalized beta (pressure/magnetic field ratio)
- `q95` - Edge safety factor
- `density` - Electron density
- `li` - Internal inductance

### Current Model: BaselineCNN

The Week 1 baseline uses a 1D Convolutional Neural Network:

```
Input: [batch, 5 channels, 1000 timesteps]
    ↓
Conv1d(5→32) + BatchNorm + ReLU + MaxPool
    ↓
Conv1d(32→64) + BatchNorm + ReLU + MaxPool
    ↓
Conv1d(64→128) + BatchNorm + ReLU + AdaptiveAvgPool
    ↓
Flatten → Linear(128→64) → ReLU → Dropout → Linear(64→2)
    ↓
Output: [batch, 2] (normal vs disruptive)
```

**Parameters:** ~81,000

### Week 2+: Fourier Neural Operators

The main innovation will be implementing FNO layers that:
- Learn operators in Fourier space for resolution-invariant predictions
- Enable transfer learning between different tokamak devices
- Incorporate physics constraints (Troyon limit, Greenwald density)

## Results (Week 1 - Synthetic Data)

| Metric | Value |
|--------|-------|
| Accuracy | 100% |
| True Positive Rate (Recall) | 100% |
| False Positive Rate | 0% |
| Precision | 100% |
| F1 Score | 1.00 |
| AUC-ROC | 1.00 |

> **Note:** These perfect metrics are expected with synthetic data where disruption patterns are clearly distinctive. Real tokamak data will present more challenging classification problems.

## Development Roadmap

| Week | Focus | Status |
|------|-------|--------|
| 1 | Data pipeline + Baseline CNN | ✅ Complete |
| 2 | Fourier Neural Operator implementation | 🔄 Next |
| 3 | Optimization + Uncertainty quantification | ⏳ Planned |
| 4 | Documentation + Final delivery | ⏳ Planned |

## 📚 References

### Neural Operators
- Li et al. (2020) "Fourier Neural Operator for Parametric PDEs" - NeurIPS
- Kovachki et al. (2021) "Neural Operator: Learning Maps Between Function Spaces" - JMLR

### Disruption Prediction
- Kates-Harbeck et al. (2019) "Predicting disruptive instabilities in controlled fusion plasmas" - Nature
- Rea et al. (2019) "Disruption prediction investigations using ML tools on DIII-D"

### Tokamak Physics
- Hender et al. (2007) "MHD stability, operational limits and disruptions" - Nuclear Fusion

## Contributing

This is a learning project, but suggestions and feedback are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

