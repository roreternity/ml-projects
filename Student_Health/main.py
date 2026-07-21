import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score

# 1. Загрузка данных
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 2. Кодирование порядковых/категориальных признаков
# Числа подобраны так, чтобы сохранить естественный порядок
# (например: low < medium < high), а не просто произвольную нумерацию
ordinal_maps = {
    "diet_type": {"veg": 1, "non-veg": 0, "balanced": 2},
    "stress_level": {"low": 1, "medium": 2, "high": 3},
    "sleep_quality": {"poor": 1, "average": 2, "good": 3},
    "physical_activity_level": {"sedentary": 1, "moderate": 2, "active": 3},
    "smoking_alcohol": {"no": 0, "occasional": 1, "yes": 2},
    "gender": {"female": 1, "male": 0, "other": 2},
}
for col, mapping in ordinal_maps.items():
    train[col] = train[col].replace(mapping)
    test[col] = test[col].replace(mapping)

# Пропуски в числовом признаке заполняем медианой по train
train["sleep_duration"] = train["sleep_duration"].fillna(train["sleep_duration"].median())
test["sleep_duration"] = test["sleep_duration"].fillna(train["sleep_duration"].median())

# 3. Признаки и целевая переменная
features = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure", "step_count",
            "exercise_duration", "water_intake", "diet_type", "stress_level",
            "sleep_quality", "physical_activity_level", "smoking_alcohol", "gender"]
X_train, y_train = train[features], train["health_condition"]
X_test = test[features]

# 4. Обучение модели
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
)
print("Обучение модели, подождите...")
model.fit(X_train, y_train)

# 5. Оценка — balanced accuracy: класс "at-risk" встречается в 86% случаев,
# обычная accuracy была бы обманчиво высокой даже для бесполезной модели
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
    print(f"  • {name:<24}: {imp:.2%}")
print("=" * 50)

# 6. Генерация сабмита
print("Генерация предсказаний для test...")
predictions = model.predict(X_test)
submission = pd.DataFrame({"id": test["id"], "health_condition": predictions})
submission.to_csv("submission.csv", index=False)
print("submission.csv успешно сохранён!")
