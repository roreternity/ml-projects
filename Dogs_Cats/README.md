# Dogs vs Cats — Image Classification

**Type:** Binary image classification (CNN)
**Competition:** [Kaggle — Dogs vs. Cats Redux](https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition)

## Task

Classify photos as a dog (1) or cat (0). Unlike MNIST, images are full-color and vary in size, so they need resizing and tensor conversion. Labels are read from the filename (`dog.123.jpg`, `cat.45.jpg`).

## Approach

1. Split `train/` filenames into train/val (80/20)
2. Resize to `128×128`, apply random horizontal flip on train (basic augmentation)
3. Train a small CNN (3 conv blocks + linear classifier)
4. Predict on `test1/`, extracting the submission `id` from the filename

## Model

**CatsDogCNN** — `Conv2d(3→16→32→64) → Linear(64·16·16 → 2)`, Adam optimizer, 5 epochs

## Files

| File | Description |
|---|---|
| `main.py` | Training + submission script |
| `dogs_cats_notebook.ipynb` | Step-by-step notebook with explanations |
