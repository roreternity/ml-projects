import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Загрузка данных
train = pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")
test = pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")

cat_features = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
drop_cols = ["Transported", "Name", "PassengerId", "Cabin"]

# 2. Признаки и целевая переменная
# Name, PassengerId, Cabin — не несут полезного сигнала в сыром виде, убираем
X = train.drop(columns=drop_cols)
y = train["Transported"].astype(int)
X_test = test.drop(columns=["Name", "PassengerId", "Cabin"])

# 3. Заполнение пропусков (train и test обрабатываем одинаково)
# -999 для чисел — CatBoost воспримет это как отдельный "сигнал отсутствия данных"
for df in [X, X_test]:
    numeric_features = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_features] = df[numeric_features].fillna(-999)
    df[cat_features] = df[cat_features].astype(str).fillna("None")

# 4. Разбиение на train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Обучение модели
# CatBoost умеет работать с категориальными признаками напрямую (cat_features),
# без ручного OneHot/LabelEncoding
model = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.05,
    depth=6,
    eval_metric="Accuracy",
    random_seed=42,
    early_stopping_rounds=50,  # останавливаемся, если качество на val перестало расти
)
model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
    verbose=100,
)

print(f"Accuracy на валидации: {accuracy_score(y_val, model.predict(X_val)):.4f}")

# 6. Генерация сабмита
test_predictions = model.predict(X_test).astype(bool)
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Transported": test_predictions,
})
submission.to_csv("submission.csv", index=False)
print("submission.csv сохранён!")
