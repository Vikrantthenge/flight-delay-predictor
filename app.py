import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import qrcode
import shap
import streamlit as st


# --- Page Setup ---
st.set_page_config(page_title="Flight Delay Predictor", layout="wide")
custom_reds = ["#4A0000", "#800000", "#8B0000", "#B22222", "#DC143C"]
MODEL_PATH = Path("model/flight_delay_model.pkl")
THRESHOLD_PATH = Path("model/decision_threshold.json")


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
**Flight Pulse** is a machine-learning dashboard that estimates whether a flight will arrive more than 15 minutes late, using airline, route, schedule, and weather inputs.  
Its saved Random Forest was evaluated on a held-out test set of 2,144 rows. The dashboard also includes illustrative airline/hour charts, model-derived feature importance, and SHAP explanations.  
Built with Streamlit for an accessible demonstration of predictive analytics.

### 🧠 Tech Stack  
Python, Pandas, Scikit-learn, Streamlit, Plotly, Joblib, SHAP, Matplotlib, qrcode

### 📈 Delay Summary  
- **Model:** Random Forest Classifier (100 trees, `class_weight="balanced"`)  
- **Test accuracy:** **72.57%**  
- **Majority-class baseline:** **67.54%**  
- **Balanced accuracy:** **67.05%**  
- **Delayed-flight precision:** approximately **59%**  
- **Delayed-flight recall:** approximately **51%**  
- **Evaluation set:** **2,144 rows**, using an 80/20 split with `random_state=42`  
- **Scope:** Historical-data demonstration; not a live operational forecast
""", unsafe_allow_html=True)


# --- Modeling Summary ---
st.markdown("## 📊 Modeling Summary")
st.markdown("""
- **Algorithm used:** Random Forest Classifier  
- **Configuration:** 100 trees with balanced class weights  
- **Verified evaluation:** 72.57% accuracy and 67.05% balanced accuracy on 2,144 held-out rows  
- **Delayed class:** Arrival delay greater than 15 minutes  
- **Prediction output:** Model-estimated delay score (0–100%), compared with the selected decision threshold  
- **Threshold:** Loaded from `model/decision_threshold.json`; falls back to 50% when that file is unavailable or invalid  

No tuned-model performance is claimed until the threshold-selected model has been measured on untouched test data.
""")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def load_delay_threshold(default=0.50):
    try:
        with THRESHOLD_PATH.open("r", encoding="utf-8") as threshold_file:
            payload = json.load(threshold_file)
        value = float(payload["delay_threshold"])
        if not 0.0 <= value <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")
        return value, True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return default, False


try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load {MODEL_PATH}. Add the trained model file before launching the app.")
    st.exception(exc)
    st.stop()

if not hasattr(model, "feature_names_in_"):
    st.error("The saved model does not contain feature_names_in_. Retrain it with a pandas DataFrame.")
    st.stop()

model_features = list(model.feature_names_in_)
delay_threshold, threshold_loaded = load_delay_threshold()


def positive_class_index(fitted_model):
    """Return the predict_proba column for the trained delayed/positive class."""
    classes = list(fitted_model.classes_)
    for candidate in (1, True, "1", "Delayed", "delay"):
        if candidate in classes:
            return classes.index(candidate)
    raise ValueError(f"The model has no recognizable delayed class. classes_={classes!r}")


try:
    delayed_index = positive_class_index(model)
except ValueError as exc:
    st.error(str(exc))
    st.stop()


# --- Input Function ---
def create_input_df(dep_hour, arr_hour, visibility, humidity, cloudcover, airline, origin, destination):
    # Start with every exact feature expected by the fitted model. This safely
    # preserves the category dropped by pd.get_dummies(..., drop_first=True).
    input_dict = {feature: 0 for feature in model_features}
    numeric_values = {
        "Dep_Hour": dep_hour,
        "Arr_Hour": arr_hour,
        "weather__hourly__visibility": visibility,
        "weather__hourly__humidity": humidity,
        "weather__hourly__cloudcover": cloudcover,
    }
    for feature, value in numeric_values.items():
        if feature in input_dict:
            input_dict[feature] = value

    # Use the exact training prefixes and set a category only if its dummy
    # column exists. A missing column is the drop-first reference category.
    for feature in (
        f"Airline_{airline}",
        f"From_{origin}",
        f"To_{destination}",
    ):
        if feature in input_dict:
            input_dict[feature] = 1

    return pd.DataFrame([input_dict], columns=model_features)


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
            unsafe_allow_html=True,
        )
        st.caption("Click to Launch")

    st.markdown("### 🔍 Model Info")
    st.markdown("""
    - **Model Used:** Random Forest (100 trees)  
    - **Test Accuracy:** 72.57%  
    - **Balanced Accuracy:** 67.05%  
    - **Delayed Precision / Recall:** ~59% / ~51%  
    - **Output:** Delay-risk score relative to the selected threshold  
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
    if origin == destination:
        st.error(
            "Origin and destination cannot be the same. "
            "Please select a different destination."
        )
        st.stop()

    input_df = create_input_df(
        dep_hour,
        arr_hour,
        visibility,
        humidity,
        cloudcover,
        airline,
        origin,
        destination,
    )
    
    delay_score = float(model.predict_proba(input_df)[0][delayed_index])
    st.metric(label="Predicted Delay Risk", value=f"{delay_score * 100:.1f}%")
    st.caption(
        f"Decision threshold: {delay_threshold * 100:.1f}%"
        + (" (loaded from model file)" if threshold_loaded else " (50% fallback)")
    )
    if delay_score >= delay_threshold:
        st.warning("Elevated delay risk: the model flags this flight for further review.")
    else:
        st.success("The model score is below the selected delay-risk threshold.")
    st.caption("This is an uncalibrated model score, not a guaranteed probability or live flight forecast.")


# --- Charts ---
airline_colors = {
    "Indigo": "#3F51B5",
    "Akasa Air": "#FFA500",
    "Air India": "#B22222",
    "Vistara": "#4B2E2E",
    "Spicejet": "#800000",
    "Air Asia": "#A52A2A",
}

st.subheader("📊 Average Delay by Airline — Illustrative Data")
st.caption("Illustrative values for dashboard demonstration; these figures are not calculated from the model evaluation set.")
airline_delay_df = pd.DataFrame({
    "Airline": ["Indigo", "Spicejet", "Air India", "Akasa Air", "Air Asia", "Vistara"],
    "Avg Delay (min)": [12, 18, 22, 15, 17, 9],
})

fig_bar = px.bar(
    airline_delay_df,
    x="Airline",
    y="Avg Delay (min)",
    color="Airline",
    title="Illustrative Average Delay by Airline",
    color_discrete_map=airline_colors,
)
st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("⏱️ Delay by Departure Hour — Illustrative Data")
st.caption("Illustrative values for dashboard demonstration; this series is not calculated from live or evaluation data.")
hourly_delay_df = pd.DataFrame({
    "Hour": list(range(24)),
    "Avg Delay (min)": [5, 6, 8, 10, 12, 15, 18, 20, 22, 19, 17, 14, 12, 10, 9, 8, 7, 6, 5, 4, 4, 5, 6, 7],
})

fig_line = px.line(
    hourly_delay_df,
    x="Hour",
    y="Avg Delay (min)",
    title="Illustrative Delay by Departure Hour",
    markers=True,
    line_shape="spline",
    color_discrete_sequence=["#8B0000"],
)
st.plotly_chart(fig_line, use_container_width=True)


# --- Feature Importance ---
st.subheader("📌 Feature Importance")
st.markdown("""
This chart shows the ten highest impurity-based feature importance scores from the loaded Random Forest.  
Importance reflects how often and how effectively a feature splits the training data; it does not show whether that feature raises or lowers delay risk.
""")
importance_df = (
    pd.DataFrame({"Feature": model_features, "Importance": model.feature_importances_})
    .sort_values("Importance", ascending=False)
    .head(10)
    .sort_values("Importance", ascending=True)
)

fig_imp = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 10 Random Forest Feature Importances",
    color_discrete_sequence=[custom_reds[3]],
)
st.plotly_chart(fig_imp, use_container_width=True)


# --- SHAP Explainability ---
st.subheader("🧠 SHAP Explainability")
st.markdown("""
The SHAP bar chart below summarizes the **mean absolute impact** of each feature across five synthetic example flights.  
Longer bars mean greater average influence on the model output for these examples; a bar chart does not show whether the influence raises or lowers delay risk.
""")

sample_inputs = pd.concat([
    create_input_df(9, 11, 5, 60, 40, "Indigo", "DEL", "BOM"),
    create_input_df(14, 16, 8, 70, 20, "Spicejet", "MAA", "HYD"),
    create_input_df(6, 8, 3, 50, 60, "Air India", "CCU", "TRV"),
    create_input_df(18, 20, 9, 85, 10, "Akasa Air", "HYD", "DEL"),
    create_input_df(22, 1, 2, 40, 90, "Vistara", "TRV", "MAA"),
], ignore_index=True)

try:
    explainer = shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(sample_inputs)

    # SHAP versions represent binary classifier output as either a list (one
    # matrix per class), a 3-D array, or a single 2-D matrix.
    if isinstance(raw_shap_values, list):
        plot_shap_values = raw_shap_values[delayed_index]
    elif getattr(raw_shap_values, "ndim", 0) == 3:
        plot_shap_values = raw_shap_values[:, :, delayed_index]
    else:
        plot_shap_values = raw_shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        plot_shap_values,
        sample_inputs,
        plot_type="bar",
        max_display=10,
        show=False,
    )
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    buf.seek(0)
    st.image(buf, use_container_width=True)
except Exception as exc:
    plt.close("all")
    st.warning("SHAP explanation could not be generated for this saved model/environment.")
    with st.expander("Technical details"):
        st.code(f"{type(exc).__name__}: {exc}")

with st.expander("🔍 What does this SHAP plot mean?"):
    st.markdown("""
    - The chart uses **five synthetic example flights**, not the training or test dataset.  
    - Bar length is the mean absolute SHAP value: the average strength of a feature's effect in those examples.  
    - A longer bar means more influence, not necessarily a higher delay risk.  
    - Direction for an individual prediction requires a directional SHAP plot, such as a waterfall plot.  
    - Results explain the loaded model; they do not establish causal relationships.
    """)


# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by **Vikrant Thenge** |")
st.caption("🕒 Last updated: October 2026")
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
