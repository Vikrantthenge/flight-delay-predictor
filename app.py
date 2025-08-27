import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
custom_reds = ["#4A0000", "#800000", "#8B0000", "#B22222", "#DC143C"]

# --- Page Setup ---
st.set_page_config(page_title="Flight Delay Predictor", layout="wide")

st.markdown(
    """
    <div style='background-color:#8B0000; padding:6px; text-align:center; border-radius:5px;'>
        <span style='color:white; font-size:16px;'>✈️ FlightPulse — Delay Forecasting Dashboard</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='text-align: center; padding: 10px 0;'>
        <h1 style='color: darkred;'>✈️ FlightPulse</h1>
        <h3 style='color: gray;'>Delay Forecasting Dashboard</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Title Section ---


st.markdown("Predict flight delays based on route, airline, and weather conditions.")

st.markdown("## 🧭 Overview")

st.markdown(
    """
    **FlightPulse** is a machine learning-powered dashboard that predicts flight delays based on airline, route, and weather conditions.  
    Built with Streamlit and trained on real aviation data, it blends predictive analytics with interactive visualizations to deliver actionable insights for travelers, airlines, and airport operations.

    ### 🔍 Key Features
    - Real-time prediction of delay probability using a trained Random Forest model  
    - Interactive input panel for flight details: departure hour, visibility, humidity, cloud cover, airline, origin, and destination  
    - Visual insights via Plotly charts:
        - 📊 Average Delay by Airline (multi-shade dark red palette)
        - ⏱️ Delay Distribution by Departure Hour (spline curve with markers)
    - Branded layout with custom banner, QR-ready thumbnail, and recruiter-polished footer  
    - Responsive design for desktop and mobile viewing

    ### 🧠 Tech Stack
    Python, Pandas, Scikit-learn, Streamlit, Plotly, Joblib, Markdown + HTML

    ### 🎯 Purpose
    To showcase predictive modeling, dashboarding, and aviation domain expertise in a visually compelling format that’s recruiter-ready and instantly accessible.
    """,
    unsafe_allow_html=True
)

# --- Load Model ---
model = joblib.load("model/flight_delay_model.pkl")

# --- Sidebar Inputs ---
st.sidebar.header("Flight Details")
dep_hour = st.sidebar.slider("Departure Hour", 0, 23, 9)
arr_hour = st.sidebar.slider("Arrival Hour", 0, 23, 11)
visibility = st.sidebar.slider("Visibility (km)", 1, 10, 5)
humidity = st.sidebar.slider("Humidity (%)", 10, 100, 60)
cloudcover = st.sidebar.slider("Cloud Cover (%)", 0, 100, 40)

airline = st.sidebar.selectbox("Airline", ["Indigo", "Spicejet", "Air India", "Go Air", "Vistara"])
origin = st.sidebar.selectbox("From", ["DEL", "BOM", "HYD", "MAA", "TRV", "CCU"])
destination = st.sidebar.selectbox("To", ["DEL", "BOM", "HYD", "MAA", "TRV", "CCU"])

# --- Prepare Input Data ---
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

# --- Prediction ---
if st.sidebar.button("Predict Delay"):
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
    color_discrete_sequence=[custom_reds[2]]  # e.g. "#8B0000"
)
st.plotly_chart(fig_line, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by Vikrant Thenge |")
st.markdown("---")

st.markdown("---")

st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)