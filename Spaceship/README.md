# Spaceship Titanic — Transported Prediction

**Type:** Binary classification with categorical features
**Competition:** [Kaggle — Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)

## Task

Predict which passengers were transported to another dimension (`Transported` — True/False) after the *Spaceship Titanic* collided with a spacetime anomaly.

## Approach

1. Drop non-predictive columns (`Name`, `PassengerId`, `Cabin`)
2. Fill missing values — numeric → `-999`, categorical → `'None'` (same treatment for train/test)
3. Train CatBoost, passing categorical features directly (no manual encoding)
4. Validate on an 80/20 stratified split with early stopping

## Model

**CatBoostClassifier** — `iterations=1500, learning_rate=0.05, depth=6, early_stopping_rounds=50`

## Files

| File | Description |
|---|---|
| `main.py` | Training + submission script |
| `spaceship_notebook.ipynb` | Step-by-step notebook with explanations |
