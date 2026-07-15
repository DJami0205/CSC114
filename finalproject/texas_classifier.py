import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# load data
df = pd.read_csv("texas_dataset.csv")

# keep only needed columns
df = df[[
    'type',
    "sub_type",
    "sqft",
    "stories",
    "beds",
    "baths",
    "garage",
    "year_built",
    "listPrice"
]].dropna()

# create price categories using tertiles
df["price_category"] = pd.qcut(
    df["listPrice"],
    q=3,
    labels=["low-priced", "mid-priced", "high-priced"]
)

# predictors and target
features = df[[
    "type",
    "sub_type",
    "sqft",
    "stories",
    "beds",
    "baths",
    "garage",
    "year_built"
]]
target = df["price_category"]

# split data 80/20
features_train, features_test, target_train, target_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42,
    stratify=target
)

# define column types
categorical_features = ["type", "sub_type"]
numeric_features = ["sqft", "stories", "beds", "baths", "garage", "year_built"]

# preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# build pipeline
model = Pipeline([
    ("preprocess", preprocessor),
    ("logreg", LogisticRegression(max_iter=1000))
])

# k-fold cross-validation on the TRAINING portion only
# (features_test / target_test stay untouched until the very end)
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

fold_scores = cross_val_score(
    model,
    features_train,
    target_train,
    cv=skf,
    scoring="accuracy"
)

print(f"{k}-Fold Cross-Validation Accuracy (on training data):")
print("Per-fold scores:", [round(s, 4) for s in fold_scores])
print("Mean accuracy:", round(fold_scores.mean(), 4))
print("Std dev:", round(fold_scores.std(), 4))
print()

# fit model on the FULL training set for the final model
model.fit(features_train, target_train)

# predictions on the held-out test set (never touched until now)
target_pred = model.predict(features_test)

# final evaluation
print("Held-out Test Set Accuracy:", round(accuracy_score(target_test, target_pred), 4))
print("\nConfusion Matrix:")
print(confusion_matrix(target_test, target_pred))
print("\nClassification Report:")
print(classification_report(target_test, target_pred))
