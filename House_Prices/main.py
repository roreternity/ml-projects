import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder

# 1. Загрузка данных
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 2. Заполнение пропусков (делаем до фичей, чтобы дальше считать без NaN)
most_frequent_zone = train["MSZoning"].mode()[0]
train["MSZoning"] = train["MSZoning"].fillna(most_frequent_zone)
test["MSZoning"] = test["MSZoning"].fillna(most_frequent_zone)

# Отсутствие подвала/гаража в данных означает "0", а не "неизвестно"
num_cols_to_fill = [
    "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath",
    "FullBath", "HalfBath", "GarageCars", "GarageArea",
]
for col in num_cols_to_fill:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)

# Кодируем категориальные колонки. unknown_value=-1 — на случай категорий,
# которых не было в train (чтобы не упасть с ошибкой на test)
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=int)
for col in ["MSZoning", "MSSubClass", "LandContour", "HouseStyle"]:
    train[f"{col}_encoded"] = encoder.fit_transform(train[[col]])
    test[f"{col}_encoded"] = encoder.transform(test[[col]])

# 3. Генерация признаков — комбинации, которые сильнее коррелируют с ценой,
# чем исходные колонки по отдельности
for df in [train, test]:
    df["House"] = df["MSSubClass"] * df["LotArea"] * df["HouseStyle_encoded"]
    df["QualityScore"] = df["OverallQual"] * df["OverallCond"]
    df["FullArea"] = df["GrLivArea"] + df["TotalBsmtSF"]
    df["BathroomScore"] = (
        df["BsmtFullBath"] + df["BsmtHalfBath"] + df["FullBath"] + df["HalfBath"]
    )
    df["GarageScore"] = df["GarageCars"] * df["GarageArea"]

# Target encoding по району: медианная цена по Neighborhood из train.
# Для районов, которых не было в train, подставляем медиану по всему train
neighborhood_price = train.groupby("Neighborhood")["SalePrice"].median()
train["NeighborhoodPrice"] = train["Neighborhood"].map(neighborhood_price)
test["NeighborhoodPrice"] = test["Neighborhood"].map(neighborhood_price)
test["NeighborhoodPrice"] = test["NeighborhoodPrice"].fillna(train["SalePrice"].median())

features = [
    "House", "QualityScore", "FullArea", "YearBuilt", "BathroomScore",
    "GarageScore", "NeighborhoodPrice", "YearRemodAdd", "LandContour_encoded",
]
X, y = train[features], train["SalePrice"]
X_test = test[features]

# 4. Обучение модели — параметры подобраны против переобучения
# (небольшая глубина деревьев, subsample/colsample < 1, min_child_weight выше дефолта)
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=42,
)
model.fit(X, y)

# 5. Оценка — CV MAE честнее, чем метрика на train (там модель просто помнит ответы)
mae_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
cv_mae = -mae_scores.mean()
y_pred = model.predict(X)

print("=" * 50)
print(" МЕТРИКИ НА TRAIN:")
print(f"  • R2:   {r2_score(y, y_pred):.4f}")
print(f"  • RMSE: {np.sqrt(mean_squared_error(y, y_pred)):,.2f} $")
print("-" * 50)
print(" КРОСС-ВАЛИДАЦИЯ (5 фолдов):")
print(f"  • MAE по фолдам: {[f'{-s:,.0f}' for s in mae_scores]}")
print(f"  • Средний CV MAE: {cv_mae:,.2f} $")
print("=" * 50)
print(" ВАЖНОСТЬ ПРИЗНАКОВ:")
for name, imp in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  • {name:<20}: {imp:.2%}")
print("=" * 50)

# 6. Генерация сабмита
test_predictions = model.predict(X_test)
submission = pd.DataFrame({"Id": test["Id"], "SalePrice": test_predictions})
submission.to_csv("submission.csv", index=False)
print("submission.csv сохранён!")
