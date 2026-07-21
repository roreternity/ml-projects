# Time Series Forecasting

**Type:** Multi-series time series forecasting  
**Dataset:** 61 monthly time series, January 2001 – February 2020  
**Forecast horizon:** 36 months (March 2020 – February 2023)

## Task

Build a forecasting model for 61 time series (US state-level data) and generate a 36-month forecast.

## Approach

### Baseline
Seasonal naive — forecast = same month last year:
```
ŷ(t) = y(t − 12)
```

### Final method — mean_delta_9
Seasonal forecast with a correction for the average year-over-year change across the last 9 same-month observations:
```
delta_12(t) = y(t) − y(t−12)
predicted_delta(t) = mean(last 9 delta_12 for this month)
ŷ(t) = y(t−12) + predicted_delta(t)
```
Window of 9 years selected by retrospective validation (tested 2–12 year windows, 9 gave best SMAPE).

### Validation
Retrospective backtest: train on data up to March 2017, forecast March 2017 – February 2020, compare against known actuals.

## Result

| Metric | mean_delta_9 | Baseline |
|---|---|---|
| MAE | 156.62 | 231.76 |
| RMSE | 356.09 | 549.33 |
| SMAPE | **18.35%** | 21.64% |

mean_delta_9 outperforms baseline on **39 of 61** series.

## Files

| File | Description |
|---|---|
| `timeline.py` | Main forecasting script |
| `notebook_timeseries.ipynb` | Step-by-step notebook |

Running the script/notebook regenerates `timeline_full.csv`, the backtest and forecast CSVs — these are excluded via `.gitignore`, not committed.
