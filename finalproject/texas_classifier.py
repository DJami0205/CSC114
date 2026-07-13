import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# load data
df = pd.read_csv("texas_dataset.csv")

# keep only needed columns
df = df[[
    "type",
    "sub_type",
    "sqft",
    "stories",
    "beds",
    "baths",
    "garage",
    "year_built",
    "listprice"
]].dropna()

# create price categories using tertiles
df["price_category"] = pd.qcut(
    df["listprice"],
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
    ("logreg", LogisticRegression(max_iter=1000, multi_class="multinomial"))
])

# fit model
model.fit(features_train, target_train)

# predictions
target_pred = model.predict(features_test)

# evaluation
print("Accuracy:", round(accuracy_score(target_test, target_pred), 4))
print("\nConfusion Matrix:")
print(confusion_matrix(target_test, target_pred))
print("\nClassification Report:")
print(classification_report(target_test, target_pred))