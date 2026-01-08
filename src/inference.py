
import pickle
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "XGB_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

def predict(input_df: pd.DataFrame):
    return model.predict(input_df)
