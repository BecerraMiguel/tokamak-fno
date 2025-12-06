# Tokamak Disruption Prediction with Fourier Neural Operators

Proof-of-concept system for predicting plasma disruptions in tokamaks using FNO.

## Project Status

Day 1: Environment setup complete

## Metrics Target

| Metric | Target |
|--------|--------|
| True Positive Rate | > 90% |
| False Positive Rate | < 10% |
| Warning Time | > 20ms |
| Inference Latency | < 10ms |

## Setup

```bash
conda activate tokamak_fno
```

## Project Structure

```
tokamak-fno/
├── src/           # Source code (models, data loaders)
├── notebooks/     # Jupyter notebooks for analysis
├── data/          # Datasets (not tracked in git)
├── results/       # Trained models and figures
├── docs/          # Documentation
├── configs/       # Configuration files
├── scripts/       # Executable scripts
└── tests/         # Unit tests
```

## Environment

-