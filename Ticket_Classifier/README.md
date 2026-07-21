# Support Ticket Classifier

**Type:** Multi-class text classification  
**Dataset:** 43,053 support ticket texts, 8 topic categories

## Task

Automatically classify support ticket texts into one of 8 topic groups: Hardware, HR Support, Access, Miscellaneous, Storage, Purchase, Internal Project, Administrative rights.

## Dataset

| Parameter | Value |
|---|---|
| Total rows | 43,053 |
| After cleaning | 43,041 |
| Classes | 8 |
| Class imbalance | Hardware 7.7× more than Administrative rights |

## Approach

1. Drop 12 rows with missing text or label
2. Stratified train/test split (80/20) — preserves class proportions
3. TF-IDF vectorization with bigrams
4. LinearSVC classifier
5. Wrapped in sklearn Pipeline (prevents data leakage)

## Model

**Pipeline: TF-IDF → LinearSVC**

| Parameter | Value | Purpose |
|---|---|---|
| `ngram_range` | (1, 2) | Unigrams + bigrams |
| `max_features` | 100,000 | Limit vector size |
| `sublinear_tf` | True | Log-scale TF |
| `min_df` | 2 | Drop very rare words |
| `C` | 1.0 | Regularization |

## Result

| Metric | Value |
|---|---|
| Accuracy | **86.79%** |
| Macro avg F1 | 86.9% |

Best class: **Purchase** (F1=0.91)  
Hardest class: **Administrative rights** (F1=0.79) — underrepresented (3.4% of data)

## Files

| File | Description |
|---|---|
| `main.py` | Training + evaluation + inference script |
| `notebook_classification.ipynb` | Step-by-step notebook |

Datasets (`*.csv`) are excluded via `.gitignore` — see the root README for how to get them.
