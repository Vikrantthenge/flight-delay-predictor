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
        elif col.startswith("TO_"):
            category = col[len("TO_"):]
            input_dict[col] = int(destination == category)

    input_df = pd.DataFrame([input_dict])
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0
    return input_df[model_features]

# --- Sidebar ---
with st.sidebar:
    st.markdown("✈️ **Flight Delay Predictor** ✈️")

    qr_url = "https://flight-delay-predictor-pulse.streamlit.app/"
    qr = qrcode.make(qr_url)
    qr_img = qr.resize((150, 150))

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(qr_img, caption="Scan to Launch", width=100)
    with col2:
        st.markdown(
            f"""
            <a href="{qr_url}" target="_blank">
                <img src="https://raw.githubusercontent.com/Vikrantthenge/flight-delay-predictor/main/thumbnail1.png" 
                     alt="Flight Pulse Delay Forecasting" width="100" style="border-radius:6px;">
            </a>
            """,
            unsafe_allow_html=True
        )
        st.caption("Click to Launch")

    st.markdown("### 🔍 Model Info")
    st.markdown("""
    - **Model Used:** Random Forest (100 trees; verify saved model)  
    - **Reported Test Accuracy:** Approximately 72%  
    - **Reported Baseline:** 67.5%  
    - **Output:** Delay probability  
    - **Mode:** Historical model with user-entered inputs  
    """)

    st.header("Flight Details")
    dep_hour = st.slider("Departure Hour", 0, 23, 9)
    arr_hour = st.slider("Arrival Hour", 0, 23, 11)
    visibility = st.slider("Visibility (km)", 1, 10, 5)
    humidity = st.slider("Humidity (%)", 10, 100, 60)
    cloudcover = st.slider("Cloud Cover (%)", 0, 100, 40)
    airline = st.selectbox("Airline", ["Indigo", "Spicejet", "Air India", "Akasa Air", "Air Asia", "Vistara"])                    
    origin = st.selectbox("From", ["DEL", "BOM", "HYD", "MAA", "TRV", "CCU"])
    destination = st.selectbox("To", ["DEL", "BOM", "HYD", "MAA", "TRV", "CCU"])
    st.caption("ℹ️ Note: 'Go Air' has been replaced with 'Akasa Air' for current relevance.")


# --- Custom Button Styling ---
st.markdown("""
    <style>
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: #d62728 !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 16px !important;
        border-radius: 6px !important;
        padding: 0.6em 2em !important;
        width: auto !important;
        margin: auto !important;
        display: block !important;
        white-space: nowrap !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #b22222 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Prediction ---
if st.sidebar.button("Predict Delay"):
    input_df = create_input_df(dep_hour, arr_hour, visibility, humidity, cloudcover, airline, origin, destination)
    probabilities = model.predict_proba(input_df)[0]
    model_classes = list(getattr(model, "classes_", []))
    if 1 in model_classes:
        delayed_index = model_classes.index(1)
    elif True in model_classes:
        delayed_index = model_classes.index(True)
    else:
        st.error(
            "Could not identify class 1 as the delayed-flight class. "
            "Please verify the target labels used during training."
        )
        st.stop()

    prediction = float(probabilities[delayed_index])
    st.metric(label="Predicted Delay Probability", value=f"{prediction*100:.1f}%")
    st.caption(
        "This is a model-estimated probability for user-entered inputs, "
        "not a live flight forecast or guaranteed outcome."
    )

# --- Charts ---
airline_colors = {
    "Indigo": "#3F51B5",        # Indigo Blue
    "Akasa Air": "#FFA500",     # Orange
    "Air India": "#B22222",     # Red
    "Vistara": "#4B2E2E",       # Dark Brown
    "Spicejet": "#800000",      # Maroon
    "Air Asia": "#A52A2A"       # Brownish Red
}

st.subheader("📊 Average Delay by Airline")
st.caption(
    "Illustrative example values hard-coded in the app, not measured "
    "from the training dataset or live airline operations."
)
airline_delay_df = pd.DataFrame({
    "Airline": ["Indigo", "Spicejet", "Air India", "Akasa Air", "Air Asia", "Vistara"],
    "Avg Delay (min)": [12, 18, 22, 15, 17, 9]
})

fig_bar = px.bar(
    airline_delay_df,
    x="Airline",
    y="Avg Delay (min)",
    color="Airline",
    title="Average Delay by Airline",
    color_discrete_map=airline_colors  # ✅ Use this instead of color_discrete_sequence
)
st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("⏱️ Delay Distribution by Departure Hour")
st.caption(
    "Illustrative example values hard-coded in the app, not measured "
    "from the training dataset or live airline operations."
)
hourly_delay_df = pd.DataFrame({
    "Hour": list(range(0, 24)),
    "Avg Delay (min)": [5, 6, 8, 10, 12, 15, 18, 20, 22, 19, 17, 14, 12, 10, 9, 8, 7, 6, 5, 4, 4, 5, 6, 7]
})

fig_line = px.line(
    hourly_delay_df,
    x="Hour",
    y="Avg Delay (min)",
    title="Delay Distribution by Departure Hour",
    markers=True,
    line_shape="spline",
    color_discrete_sequence=["#8B0000"]  # Optional: pick a single tone from your reds
)
st.plotly_chart(fig_line, use_container_width=True)

# --- Feature Importance ---
st.subheader("📌 Feature Importance")
st.markdown("""
This chart shows the top 10 feature-importance scores exposed by the loaded model.  
Importance indicates how the model uses features; it does not prove that a feature causes flight delays.
""")
importance_df = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
})
 
fig_imp = px.bar(
    importance_df.head(10),
    x="Feature",
    y="Importance",
    title="Top 10 Feature Importances",
    color_discrete_sequence=[custom_reds[3]]
)
st.plotly_chart(fig_imp, use_container_width=True)

# --- SHAP Explainability ---
st.subheader("🧠 SHAP Explainability")
st.markdown("""
SHAP (SHapley Additive exPlanations) helps interpret how features contribute to model predictions.  
The summary below uses five example inputs; it does not explain the current user's prediction.
""")

# Create sample inputs
sample_inputs = pd.concat([
    create_input_df(9, 11, 5, 60, 40, "Indigo", "DEL", "BOM"),
    create_input_df(14, 16, 8, 70, 20, "Spicejet", "MAA", "HYD"),
    create_input_df(6, 8, 3, 50, 60, "Air India", "CCU", "TRV"),
    create_input_df(18, 20, 9, 85, 10, "Akasa Air", "HYD", "DEL"),
    create_input_df(22, 1, 2, 40, 90, "Vistara", "TRV", "MAA")
])

# Generate SHAP values. Compatibility can vary by model and SHAP version.
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_inputs)

    # Select the positive/delayed class for common binary classifier outputs.
    if isinstance(shap_values, list):
        plot_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        plot_values = shap_values
        if getattr(plot_values, "ndim", 0) == 3:
            plot_values = plot_values[:, :, -1]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        plot_values,
        sample_inputs,
        plot_type="bar",
        show=False
    )
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    st.image(buf)
except Exception as exc:
    st.warning(
        "SHAP could not generate a plot for the loaded model. "
        f"Check model compatibility and SHAP version. Details: {exc}"
    )

# Layman-friendly explanation
with st.expander("🔍 What does this SHAP plot mean?"):
    st.markdown("""
    - **Longer bars** = Greater average absolute SHAP impact across the example inputs.
    - This bar plot shows impact magnitude, not whether a feature increases or decreases delay risk.
    - Only five example rows are used, so the ranking may be unstable.
    - SHAP explains model behavior; it does not establish the true causes of flight delays.
    """)

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by **Vikrant Thenge** |")
st.caption("🕒 Last updated: September 2025")
st.markdown("---")

st.markdown("""
<div style='text-align: center; font-size: 16px;'>
    <p><strong style='color: darkred;'>Crafted with precision. Powered by data. Designed for impact.</strong></p>
    <p>
        <a href='https://github.com/vikrantthenge' target='_blank'>
            <img src='https://cdn-icons-png.flaticon.com/512/25/25231.png' width='20' style='vertical-align:middle;' />
            <span style='margin-left:8px;'>GitHub</span>
        </a> &nbsp;&nbsp;&nbsp;
        <a href='https://linkedin.com/in/vthenge' target='_blank'>
            <img src='https://cdn-icons-png.flaticon.com/512/174/174857.png' width='20' style='vertical-align:middle;' />
            <span style='margin-left:8px;'>LinkedIn</span>
        </a> &nbsp;&nbsp;&nbsp;
        <a href='mailto:vikrantthenge@outlook.com'>
            <img src='https://cdn-icons-png.flaticon.com/512/732/732223.png' width='20' style='vertical-align:middle;' />
            <span style='margin-left:8px;'>Outlook</span>
        </a>
    </p>
</div>
""", unsafe_allow_html=True)
