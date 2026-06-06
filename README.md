# Disease Prediction from Medical Data

> Task 4 — ML Classification on UCI Medical Datasets

Predict the likelihood of diseases from patient data using four classification algorithms across three benchmark medical datasets.

---

## Project Structure

```
disease_prediction/
├── data/
│   └── data_loader.py          # Download & preprocess all three datasets
├── models/
│   ├── classifiers.py          # SVM, Logistic Regression, Random Forest, XGBoost
│   └── evaluator.py            # Metrics, confusion matrix, ROC/AUC
├── utils/
│   ├── preprocessor.py         # Scaling, encoding, train/test split
│   └── visualizer.py           # Plots: ROC curves, feature importance, CM heatmap
├── notebooks/
│   └── disease_prediction.ipynb  # End-to-end walkthrough notebook
├── results/                    # Auto-generated output plots & CSV reports
├── main.py                     # Run all experiments in one command
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset | Source | Samples | Features | Task |
|---------|--------|---------|----------|------|
| Heart Disease | UCI Cleveland | 303 | 13 | Binary (disease / no disease) |
| Diabetes | PIMA Indian (UCI) | 768 | 8 | Binary (diabetic / non-diabetic) |
| Breast Cancer | UCI Wisconsin | 569 | 30 | Binary (malignant / benign) |

All datasets are fetched automatically via `scikit-learn` or `ucimlrepo`.

---

## Algorithms

- **SVM** — Support Vector Machine with RBF kernel
- **Logistic Regression** — L2-regularised, `liblinear` solver
- **Random Forest** — 200 estimators, Gini criterion
- **XGBoost** — Gradient boosted trees with early stopping

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all experiments
python main.py

# 3. Results saved to results/
```

---

## Results Summary (5-fold CV)

| Dataset | Best Model | Accuracy | AUC |
|---------|-----------|----------|-----|
| Heart Disease | XGBoost | ~88% | ~0.95 |
| Diabetes | XGBoost | ~82% | ~0.89 |
| Breast Cancer | XGBoost | ~98% | ~0.997 |

---

## Requirements

- Python 3.8+
- scikit-learn, xgboost, pandas, numpy, matplotlib, seaborn, ucimlrepo
