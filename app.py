import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import qrcode
from PIL import Image

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

st.markdown("Predict flight delays based on route, airline, and weather conditions.")

st.markdown("""
<div style='background-color:#8B0000; padding:6px; text-align:center; border-radius:5px;'>
    <span style='color:white; font-size:16px;'> Delay Forecasting Dashboard</span>
</div>
""", unsafe_allow_html=True)

st.markdown("## 🧭 Overview")
st.markdown("""
**Flight Pulse** is a machine learning-powered dashboard that predicts flight delays based on airline, route, and weather conditions.  
Built with Streamlit and trained on real aviation data, it blends predictive analytics with interactive visualizations to deliver actionable insights for travelers, airlines, and airport operations.  

### 🧠 Tech Stack
Python, Pandas, Scikit-learn, Streamlit, Plotly, Joblib, Markdown + HTML, qrcode, PIL (Pillow)  

### 🎯 Purpose
To showcase predictive modeling, dashboarding, and aviation domain expertise in a visually compelling format that is instantly accessible.
""", unsafe_allow_html=True)

# --- Load Model ---
model = joblib.load("model/flight_delay_model.pkl")

# --- Function for Inputs ---
def create_input_df(dep_hour, arr_hour, visibility, humidity, cloudcover, airline, origin, destination):
    input_dict = {
        "Dep_Hour": dep_hour,
        "Arr_Hour": arr_hour,
        "weather__hourly__visibility": visibility,
        "weather__hourly__humidity": humidity,
        "weather__hourly__cloudcover": cloudcover
    }

    for a in ["Air India", "Go Air", "Indigo", "Spicejet", "Vistara"]:
        input_dict[f"Airline_{a}"] = 1 if airline == a else 0
    for o in ["BOM", "CCU", "DEL", "HYD", "MAA", "TRV"]:
        input_dict[f"From_{o}"] = 1 if origin == o else 0
    for d in ["BOM", "CCU", "DEL", "HYD", "MAA", "TRV"]:
        input_dict[f"TO_{d}"] = 1 if destination == d else 0

    input_df = pd.DataFrame([input_dict])

    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model.feature_names_in_]
    return input_df

# --- Sidebar Layout ---
with st.sidebar:
    st.markdown("✈️ **Flight Delay Predictor** ✈️")

    # QR Code + Thumbnail
    qr_url = "https://flight-delay-predictor-pulse.streamlit.app/"
    qr = qrcode.make(qr_url)
    qr_img = qr.resize((150, 150))

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(qr_img, caption="Scan to Launch App", width=100)
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
        st.caption("Flight Pulse Delay Forecasting")

    st.header("Flight Details")
    dep_hour = st.slider("Departure Hour", 0, 23, 9, key="dep_hour")
    arr_hour = st.slider("Arrival Hour", 0, 23, 11, key="arr_hour")
    visibility = st.slider("Visibility (km)", 1, 10, 5, key="visibility")
    humidity = st.slider("Humidity (%)", 10, 100, 60, key="humidity")
    cloudcover = st.slider("Cloud Cover (%)", 0, 100, 40, key="cloudcover")

    airline = st.selectbox("Airline", ["Indigo", "Spicejet", "Air India", "Go Air", "Vistara"], key="airline")
    origin = st.selectbox("From", ["DEL", "BOM", "HYD", "MAA", "TRV", "CCU"], key="origin")
    destination = st.selectbox("To", ["DEL", "BOM", "HYD", "MAA", "TRV", "CCU"], key="destination")

# --- Custom CSS for Centered Red Button ---
st.markdown("""
    <style>
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: #d62728 !important;   /* Red */
        color: white !important;
        font-weight: bold !important;
        font-size: 16px !important;
        border-radius: 6px !important;
        padding: 0.6em 2em !important;          /* Horizontal padding for width */
        width: auto !important;                 /* Fit to content */
        margin: auto !important;                /* Center it */
        display: block !important;
        white-space: nowrap !important;         /* Prevent wrapping */
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #b22222 !important;   /* Dark red hover */
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)


# --- Prediction Button ---
if st.sidebar.button("Predict Delay", key="predict_btn"):
    input_df = create_input_df(dep_hour, arr_hour, visibility, humidity, cloudcover, airline, origin, destination)
    prediction = model.predict_proba(input_df)[0][1]
    st.metric(label="Predicted Delay Probability", value=f"{prediction*100:.1f}%")


# --- Charts ---
st.subheader("📊 Average Delay by Airline")
airline_delay_df = pd.DataFrame({
    "Airline": ["Indigo", "Spicejet", "Air India", "Go Air", "Vistara"],
    "Avg Delay (min)": [12, 18, 22, 15, 9]
})
fig_bar = px.bar(
    airline_delay_df,
    x="Airline",
    y="Avg Delay (min)",
    color="Airline",
    title="Average Delay by Airline",
    color_discrete_sequence=custom_reds
)
st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("⏱️ Delay Distribution by Departure Hour")
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
    color_discrete_sequence=[custom_reds[2]]
)
st.plotly_chart(fig_line, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by **Vikrant Thenge** |")
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
