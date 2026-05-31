# Anomaly Detection with Autoencoders on Credit Card Fraud

An **autoencoder** is trained only on legitimate transactions, and its
**reconstruction error** is used as an anomaly score. The model is
compared against three baselines (Isolation Forest, One-Class SVM, and a Gaussian
density model) on the *Credit Card Fraud Detection* dataset and evaluated
with metrics suited to extreme class imbalance (PR-AUC and ROC-AUC).

## Requirements

- Python >= 3.11
- Virtual environment via `venv`; project configuration in `pyproject.toml`

## Setup
1. Create and activate the virtual environment
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```
2. Install the project
```bash
python -m pip install --upgrade pi
python -m pip install -e .
```