import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import qrcode
import shap
import matplotlib.pyplot as plt
import io

# --- Page Setup ---
st.set_page_config(page_title="Flight Delay Predictor", layout="wide")
custom_reds = ["#4A0000", "#800000", "#8B0000", "#B22222", "#DC143C"]

# --- Header ---
st.markdown("""
<div style='background-color:#8B0000; padding:6px; text-align:center; border-radius:5px;'>
    <span style='color:white; font-size:16px;'> Flight Pulse </span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; padding: 10px 0;'>
    <h1 style='color: darkred;'>✈️ Flight Pulse</h1>
    <h3 style='color: gray;'>Delay Forecasting Dashboard</h3>
</div>
""", unsafe_allow_html=True)

# --- Overview ---
st.markdown("## 🧭 Overview")
st.markdown("""
**Flight Pulse** is a machine learning-powered dashboard that predicts flight delays based on airline, route, and weather conditions.  
Uses a previously trained Random Forest classifier to estimate flight delay risk from airline, route, schedule, and weather inputs.  
Includes model feature-importance and SHAP visualizations.  
Visualizes illustrative delay patterns by airline and departure hour using Plotly charts.  
This is a demonstration app and does not use live flight or weather feeds.

### 🧠 Tech Stack  
Python, Pandas, Scikit-learn, Streamlit, Plotly, Joblib, Markdown + HTML, qrcode, PIL (Pillow)

### 📈 Delay Summary  
- Training dataset: 10K+ historical flight records (reported; verify source)  
- Model used: **Random Forest (100 trees)**  
- Reported test accuracy: **approximately 72%**  
- Reported majority-class baseline: **67.5%**  
- Evaluation figures must be verified against the training script  
- Prediction output: model-estimated delay probability  
""", unsafe_allow_html=True)

# --- Modeling Summary ---
st.markdown("## 📊 Modeling Summary")
st.markdown("""
- **Algorithm:** Random Forest (100 trees; confirm against saved model)  
- **Training Volume:** 10K+ historical flight records (reported)  
- **Reported Test Accuracy:** Approximately 72%  
- **Reported Baseline:** 67.5% (always predict not delayed)  
- **Evaluation:** See the training script for the actual test split and metrics  
- **Prediction Output:** Model-estimated delay probability (0–100%)  
- **Deployment:** Interactive demonstration, not live operational forecasting  
""")

# --- Load Model ---
model = joblib.load("model/flight_delay_model.pkl")

# --- Input Function ---
def create_input_df(dep_hour, arr_hour, visibility, humidity, cloudcover, airline, origin, destination):
    input_dict = {
        "Dep_Hour": dep_hour,
        "Arr_Hour": arr_hour,
        "weather__hourly__visibility": visibility,
        "weather__hourly__humidity": humidity,
        "weather__hourly__cloudcover": cloudcover
    }
    # Encode categories using the feature names expected by the saved model.
    # This also prevents selectable categories such as Air Asia from being
    # silently omitted when the saved model contains a corresponding feature.
    model_features = list(model.feature_names_in_)

    for col in model_features:
        if col.startswith("Airline_"):
            category = col[len("Airline_"):]
            input_dict[col] = int(airline == category)
        elif col.startswith("From_"):
            category = col[len("From_"):]
            input_dict[col] = int(origin == category)
        elif col.startswith("To_"):
            category = col[len("TO_"):]
            input_dict[col] = int(destination == category)

    input_df = pd.DataFrame([input_dict])
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0
    return input_df[model_features]

# --- Sidebar ---
with st.sidebar:
