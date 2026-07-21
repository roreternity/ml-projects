# Guardian Data Collection

**Type:** Web data collection via REST API  
**Source:** The Guardian Open Platform

## Task

Collect a dataset of 5,000 news articles from The Guardian via their official Content API, extract text features and compute readability/sentiment metrics.

## Approach

1. Authenticate via Guardian API key
2. Paginate through results (50 articles/page)
3. Extract fields: title, author, topic, body text, tags
4. Compute per-article metrics:
   - **Word count** — raw word count from body text
   - **Readability** — Flesch-Kincaid Grade Level (computed from scratch)
   - **Sentiment** — positive/neutral/negative via keyword scoring
   - **Links** — extracted from HTML body via custom parser
   - **Entities** — from Guardian keyword tags

## Dataset

| Field | Description |
|---|---|
| `date` | Publication date |
| `topic` | Section (e.g. World news, Sport) |
| `title` | Article headline |
| `author` | Byline |
| `entities` | Keyword tags |
| `word_count` | Number of words |
| `readability_score` | Flesch-Kincaid score |
| `readability_level` | easy / medium / hard / very hard |
| `sentiment` | positive / neutral / negative |
| `sentiment_score` | Raw score (positive − negative word count) |
| `url` | Article URL |

**Result:** 5,000 articles, 12 fields, saved as `guardian_5000.tsv`

## Key Finding

US news and World news sections have negative average sentiment; Football and Sport have positive — showing the method captures real signal.

## Files

| File | Description |
|---|---|
| `collect_guardian_dataset.py` | Main collection script |
| `analyze_guardian_dataset.py` | Dataset analysis script |
| `notebook_guardian_collect.ipynb` | Step-by-step notebook |

Running `collect_guardian_dataset.py` (with a `GUARDIAN_API_KEY` env var set) regenerates `guardian_5000.tsv`.
