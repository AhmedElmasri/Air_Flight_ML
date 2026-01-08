# ✈️ Air Flight Price Prediction

This project is a **machine learning application** that predicts **flight ticket prices** based on flight details such as airline, source, destination, number of stops, and journey time.

The project includes:
- Data preprocessing
- Model training using **XGBoost**
- Metadata generation for inference
- A **Streamlit web application** for interactive predictions

---

## 📁 Project Structure

```
Air Flight/
│
├── Data/
│   └── clean/
│       └── cleaned_flight_data.parquet
│
├── metadata/
│   ├── inputs.pkl
│   └── unique_values_dict.pkl
│
├── models/
│   └── XGB_model.pkl
│
├── Notebook/
│   └── Air_Flight_ML.ipynb
│
├── src/
│   ├── app.py
│   ├── train.py
│   ├── inference.py
│   ├── evaluate.py
│   ├── preprocessing.py
│   └── __init__.py
│
└── README.md
```

---

## ⚙️ Requirements

Install the required Python packages:

```bash
pip install pandas scikit-learn xgboost pyarrow streamlit
```

---

## 🚀 How to Run the Project

### 1️⃣ Train the model and generate metadata

From the project root directory:

```bash
python src/train.py
```

This will create:
- `models/XGB_model.pkl`
- `metadata/inputs.pkl`
- `metadata/unique_values_dict.pkl`

---

### 2️⃣ Run the Streamlit application

```bash
python -m streamlit run src/app.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

## 🧠 Model Details

- **Model**: XGBoost Regressor
- **Target Variable**: `Price`
- **Evaluation Metrics**:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)

---

## 🖥️ Streamlit App Features

- Dropdowns for categorical inputs (Airline, Source, Destination)
- Numeric inputs for journey details
- Instant flight price prediction

---

## 📌 Notes

- Ensure all dependencies are installed in the same Python environment.
- The metadata files are automatically generated during training.
- The project is structured for easy extension and deployment.

---

## 📄 License

This project is for educational and learning purposes.
