# Titanic — Survival Prediction

**Type:** Binary classification  
**Competition:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

## Task

Predict which passengers survived the Titanic shipwreck based on passenger data (age, sex, class, etc.).

## Approach

1. Fill missing values (Age → median, Embarked → mode)
2. Encode categorical features (Sex, Embarked)
3. Train a Random Forest classifier with cross-validation

## Model

**Random Forest** — `n_estimators=300, max_depth=5, min_samples_split=5`

## Result

~79% accuracy (5-fold cross-validation)

## Files

| File | Description |
|---|---|
| `main.py` | Training + cross-validation + submission script |
| `titanic_notebook.ipynb` | Step-by-step notebook with explanations |
