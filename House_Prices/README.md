# House Prices — Advanced Regression

**Type:** Regression  
**Competition:** [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## Task

Predict the final sale price of homes in Ames, Iowa based on 79 features (size, quality, neighborhood, age, etc.).

## Approach

1. Fill missing values (numeric → 0, categorical → mode)
2. Encode categorical features with OrdinalEncoder
3. Feature engineering — composite features:
   - `FullArea` = GrLivArea + TotalBsmtSF
   - `QualityScore` = OverallQual × OverallCond
   - `BathroomScore` = sum of all bathrooms
   - `GarageScore` = GarageCars × GarageArea
4. Target encoding for Neighborhood (median sale price per neighborhood)
5. Train XGBoost with anti-overfitting parameters

## Model

**XGBoost** — `n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8`

## Result

| Metric | Value |
|---|---|
| Cross-val MAE | ~$16,500 |
| Train R² | 0.967 |

## Feature Importance

| Feature | Importance |
|---|---|
| FullArea | 30% |
| NeighborhoodPrice | 22% |
| BathroomScore | 12% |
| QualityScore | 11% |
| GarageScore | 10% |

## Files

| File | Description |
|---|---|
| `main.py` | Training + submission script |
| `house_prices_notebook.ipynb` | Step-by-step notebook with explanations |
