
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Data", "clean", "cleaned_flight_data.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

df = pd.read_parquet(DATA_PATH)

TARGET = "Price"

NUMERIC_FEATURES = [
    "Total_Stops",
    "Duration_minutes",
    "Journey_Day",
    "Journey_Month",
    "Dep_Hour",
    "Arrival_Hour"
]

CATEGORICAL_FEATURES = [
    "Airline",
    "Source",
    "Destination"
]

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]

# =========================
# SAVE METADATA FILES
# =========================
inputs = {
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES
}

with open(os.path.join(METADATA_DIR, "inputs.pkl"), "wb") as f:
    pickle.dump(inputs, f)

unique_values_dict = {
    col: sorted(df[col].dropna().unique().tolist())
    for col in CATEGORICAL_FEATURES
}

with open(os.path.join(METADATA_DIR, "unique_values_dict.pkl"), "wb") as f:
    pickle.dump(unique_values_dict, f)

# =========================
# MODEL PIPELINE
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
    ]
)

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

with open(os.path.join(MODEL_DIR, "XGB_model.pkl"), "wb") as f:
    pickle.dump(pipeline, f)

print("Training finished successfully.")
