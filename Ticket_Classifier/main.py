import sys
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "public_data_csv.csv")
INPUT_PATH = os.path.join(BASE_DIR, "input.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "output.csv")
TARGET_COL = "Topic_group"
TEXT_COL = "Document"
ID_COL = "Row_id"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 1. Загрузка и очистка данных
print("Загрузка датасета:", DATASET_PATH)
df = pd.read_csv(DATASET_PATH)
print(f"Загружено строк: {len(df)}")

# Строки с пропусками удаляем, а не заполняем — заполнить целевой класс нечем
df = df.dropna(subset=[TEXT_COL, TARGET_COL])
print(f"После удаления пропусков: {len(df)}")

print("\nРаспределение классов:")
print(df[TARGET_COL].value_counts().to_string())

# 2. Разбиение на train/test (со стратификацией — классы несбалансированы)
x = df[TEXT_COL].fillna("").astype(str)
y = df[TARGET_COL]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"\nРазбиение: train={len(x_train)}, test={len(x_test)}")

# 3. Построение и обучение pipeline — TF-IDF векторизатор + LinearSVC
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),   # униграммы + биграммы
        max_features=100000,  # ограничение размерности словаря
        sublinear_tf=True,    # логарифмируем частоту слова
        min_df=2,             # выкидываем слова, встречающиеся 1 раз
    )),
    ("clf", LinearSVC(
        C=1.0,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )),
])

pipeline.fit(x_train, y_train)
print("Обучение завершено")

# 4. Оценка на тестовой выборке
y_pred = pipeline.predict(x_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nAccuracy на тесте: {acc:.4f} ({acc*100:.2f}%)")
print("\nОтчёт по классам:")
print(classification_report(y_test, y_pred))

# 5. Классификация input.csv, если он есть рядом со скриптом
if not os.path.exists(INPUT_PATH):
    print(f"\nФайл {INPUT_PATH} не найден. Завершение работы.")
    print("Поместите input.csv рядом с main.py, чтобы классифицировать новые обращения.")
    sys.exit(0)

print(f"\nЧтение входного файла: {INPUT_PATH}")
df_input = pd.read_csv(INPUT_PATH)
print(f"Строк в input.csv: {len(df_input)}")

X_input = df_input[TEXT_COL].fillna("").astype(str)
df_input[TARGET_COL] = pipeline.predict(X_input)

df_input.to_csv(OUTPUT_PATH, index=False)
print(f"   Результат сохранён в: {OUTPUT_PATH}")
print("\nПример предсказаний:")
print(df_input[[ID_COL, TARGET_COL]].head(10).to_string(index=False))
print("\nГотово!")
