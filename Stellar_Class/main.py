import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# 1. Загрузка данных
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 2. Кодирование категориальных признаков
# LabelEncoder обучаем на train и применяем на test тем же самым энкодером —
# иначе одинаковые категории могут получить разные коды
le_spec = LabelEncoder()
train["spectral_type_encoded"] = le_spec.fit_transform(train["spectral_type"])
test["spectral_type_encoded"] = le_spec.transform(test["spectral_type"])

le_gal = LabelEncoder()
train["galaxy_population_encoded"] = le_gal.fit_transform(train["galaxy_population"])
test["galaxy_population_encoded"] = le_gal.transform(test["galaxy_population"])

# 3. Признаки и целевая переменная
features = ["alpha", "delta", "u", "g", "r", "i", "z",
            "redshift", "spectral_type_encoded", "galaxy_population_encoded"]
X_train, y_train = train[features], train["class"]
X_test = test[features]

# 4. Обучение модели
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,        # классы (GALAXY/QSO/STAR) разделяются сложными нелинейными границами
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
)
print("Обучение модели, подождите...")
model.fit(X_train, y_train)

# 5. Оценка — balanced accuracy, т.к. классы несбалансированы
print("Запуск кросс-валидации...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="balanced_accuracy", n_jobs=-1)
train_acc = balanced_accuracy_score(y_train, model.predict(X_train))

print("=" * 50)
print(" МЕТРИКИ (Balanced Accuracy):")
print(f"  • Train:                 {train_acc:.4f}")
print(f"  • Кросс-валидация (CV):  {cv_scores.mean():.4f}")
print("=" * 50)
print(" ВАЖНОСТЬ ПРИЗНАКОВ (по убыванию):")
for name, imp in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  • {name:<28}: {imp:.2%}")
print("=" * 50)

# 6. Генерация сабмита
print("Генерация предсказаний для test...")
predictions = model.predict(X_test)
submission = pd.DataFrame({"id": test["id"], "class": predictions})
submission.to_csv("submission.csv", index=False)
print("submission.csv успешно сохранён!")
