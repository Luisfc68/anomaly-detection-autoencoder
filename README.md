# Anomaly Detection with Autoencoders — Credit Card Fraud

A complete unsupervised Anomaly Detection pipeline using Autoencoders applied to the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle).

## Setup

**1. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

**2. Install the project**

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

## Usage

From the root of the repository:

```bash
python src/fraud/main.py
```

Evaluation curves (ROC, PR) and exploratory data figures are marginsaved to the `results/figures/` directory. Console output includes model comparison metrics against baselines, bootstrapped 95% confidence intervals, and detailed performance reports across F1-maximizing, high-recall, and high-precision operational thresholds.

