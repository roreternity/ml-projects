import pandas as pd

# Быстрый анализ собранного датасета: размер, распределения, пропуски
df = pd.read_csv("guardian_5000.tsv", sep="\t")

print(df.shape)
print(df["word_count"].describe())
print(df["topic"].value_counts().head(10))
print(df["sentiment"].value_counts())
print(df.groupby("topic")[["readability_score", "sentiment_score"]].mean())
print(df.isna().sum())
