import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# 1. Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 2. Handle missing values
for df in [train, test]:
    df["Age"] = df["Age"].fillna(train["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(train["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(train["Fare"].median())

# 3. Encode categorical features
for df in [train, test]:
    df["Sex"] = df["Sex"].replace({"female": 1, "male": 0})
    df["Embarked"] = df["Embarked"].replace({"S": 1, "C": 2, "Q": 3})

# 4. Select features and train
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X, y = train[features], train["Survived"]
X_test = test[features]

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
)
model.fit(X, y)

# 5. Cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print("Accuracy per fold:", scores.round(4))
print(f"Mean accuracy:     {scores.mean():.4f}")

# 6. Generate submission
predictions = model.predict(X_test)
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})
submission.to_csv("submission.csv", index=False)
print("submission.csv saved!")
