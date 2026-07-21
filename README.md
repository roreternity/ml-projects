# ML Portfolio — Kaggle & University Practice

A collection of machine learning projects covering classification, regression, computer vision, NLP, time series forecasting, web scraping, and LLM fine-tuning.

Each project folder has a `README.md` with the task description and a `*_notebook.ipynb` with a full Russian-language walkthrough (code and output stay in English).

---

## Projects

### Kaggle Competitions

| Project | Task | Model | Result |
|---|---|---|---|
| [Titanic](./Titanic/) | Binary classification — survival prediction | Random Forest | ~79% accuracy |
| [House Prices](./House_Prices/) | Regression — sale price prediction | XGBoost | MAE ~$16,500 |
| [Spaceship Titanic](./Spaceship/) | Binary classification — categorical features | CatBoost | — |
| [Digit Recognizer](./Digit_Recognizer/) | Image classification (MNIST) | CNN (PyTorch) | — |
| [Dogs vs Cats](./Dogs_Cats/) | Binary image classification | CNN (PyTorch) | — |
| [Stellar Object Classification](./Stellar_Class/) | Multi-class classification (SDSS) | Random Forest | — |
| [Student Health Classification](./Student_Health/) | Multi-class, imbalanced classes | Random Forest | — |

### University Practice

| Project | Task | Method | Result |
|---|---|---|---|
| [Guardian Data Collection](./News_Parser/) | Web data collection via API | Guardian Content API | 5,000 articles, 12 features |
| [Support Ticket Classifier](./Ticket_Classifier/) | Multi-class text classification | TF-IDF + LinearSVC | 86.79% accuracy (8 classes) |
| [Time Series Forecasting](./Energy_Forecasting/) | Multi-series forecasting (61 series, 36 months) | Seasonal mean-delta | SMAPE 18.35% vs 21.64% baseline |
| [Least Squares & Linear Regression](./Study/) | Curve fitting, regression fundamentals | numpy / sklearn / scipy | — |

### Other

| Project | Description |
|---|---|
| [DojoTask](./DojoTask/) | EN→RU translation with Gemma + QLoRA fine-tuning |
| [MII](./MII/) | Web scraping scripts (HH.ru vacancies, Wildberries products) |

---

## Data

Datasets and generated CSV outputs are excluded from this repository via `.gitignore` (Kaggle competition data shouldn't be redistributed, and large files don't belong in git). Each project's notebook expects the corresponding Kaggle dataset in `/kaggle/input/...` when run on Kaggle, or the raw CSVs placed next to the script when run locally.

## Stack

Python · pandas · scikit-learn · XGBoost · CatBoost · PyTorch · NumPy
