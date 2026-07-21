# Stellar Object Classification (SDSS)

**Type:** Multi-class classification
**Dataset:** Sloan Digital Sky Survey photometric data

## Task

Classify sky objects into `GALAXY`, `QSO`, or `STAR` based on photometric features (magnitudes, redshift, spectral type).

## Approach

1. Encode categorical features (`spectral_type`, `galaxy_population`) with `LabelEncoder`
2. Train a Random Forest classifier
3. Evaluate with balanced accuracy (classes are imbalanced) and 5-fold cross-validation

## Model

**Random Forest** — `n_estimators=300, max_depth=15, min_samples_split=5`

## Files

| File | Description |
|---|---|
| `main.py` | Training + evaluation + submission script |
| `stellar_notebook.ipynb` | Step-by-step notebook with explanations |
