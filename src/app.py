
import streamlit as st
import pickle
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "metadata", "inputs.pkl"), "rb") as f:
    inputs = pickle.load(f)

with open(os.path.join(BASE_DIR, "metadata", "unique_values_dict.pkl"), "rb") as f:
    unique_vals = pickle.load(f)

with open(os.path.join(BASE_DIR, "models", "XGB_model.pkl"), "rb") as f:
    model = pickle.load(f)

st.title("Flight Price Prediction")

user_input = {}

for col in inputs["categorical_features"]:
    user_input[col] = st.selectbox(col, unique_vals[col])

for col in inputs["numeric_features"]:
    user_input[col] = st.number_input(col, value=0.0)

if st.button("Predict Price"):
    df = pd.DataFrame([user_input])
    price = model.predict(df)[0]
    st.success(f"Estimated Flight Price: {price:.2f}")
