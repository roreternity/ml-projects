# Student Health Condition Classification

**Type:** Multi-class classification with imbalanced classes

## Task

Predict a student's health condition from lifestyle features (sleep, diet, activity, stress, etc.).

## Approach

1. Encode ordinal/categorical features (`diet_type`, `stress_level`, `sleep_quality`, `physical_activity_level`, `smoking_alcohol`, `gender`)
2. Fill missing `sleep_duration` with the train median
3. Train a Random Forest classifier
4. Evaluate with balanced accuracy and 5-fold cross-validation (the target class is imbalanced — at-risk ~86%)

## Model

**Random Forest** — `n_estimators=300, max_depth=10, min_samples_split=5`

## Files

| File | Description |
|---|---|
| `main.py` | Training + evaluation + submission script |
| `student_health_notebook.ipynb` | Step-by-step notebook with explanations |
