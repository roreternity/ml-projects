# Digit Recognizer — MNIST Classification

**Type:** Image classification (CNN)
**Competition:** [Kaggle — Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer)

## Task

Classify handwritten digits (0–9) from 28×28 grayscale images, provided as flattened pixel rows in CSV.

## Approach

1. Normalize pixels to `[0, 1]` and reshape each row to `(1, 28, 28)` for `Conv2d`
2. Split the labeled `train.csv` into train/val (80/20) — Kaggle's `test.csv` has no labels
3. Train a small CNN (2 conv blocks + linear classifier)
4. Track train/val accuracy each epoch to watch for overfitting

## Model

**SimpleCNN** — `Conv2d(1→16) → Conv2d(16→32) → Linear(32·7·7 → 10)`, Adam optimizer, 5 epochs

## Files

| File | Description |
|---|---|
| `main.py` | Training + submission script |
| `digit_recognizer_notebook.ipynb` | Step-by-step notebook with explanations |
