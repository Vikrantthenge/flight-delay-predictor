# ✈️ Flight Pulse — Flight Delay Prediction Dashboard

<p align="center">
  <img src="https://github.com/Vikrantthenge/flight-delay-predictor/blob/main/thumbnail.png" alt="FlightPulse Thumbnail" width="150"/>
</p>

**FlightPulse** is a machine learning-powered dashboard that predicts flight delays based on airline, route, and weather conditions. Built with Streamlit and trained on real aviation data, it blends predictive analytics with interactive visualizations to deliver actionable insights for travelers, airlines, and airport operations.

---

## 🔍 Features

- 🎯 Real-time delay prediction using a trained Random Forest model  
- 🧭 Interactive input panel for flight details: departure hour, visibility, humidity, cloud cover, airline, origin, and destination  
- 📊 Visual insights via Plotly charts:
  - Average Delay by Airline (multi-shade dark red palette)
  - Delay Distribution by Departure Hour (spline curve with markers)
- 🧮 SHAP-based feature importance plots for model explainability  
- 🎨 Branded layout with custom banner, QR-ready thumbnail, and recruiter-polished footer  
- 📱 Responsive design for desktop and mobile viewing

---

## 🧠 Tech Stack

| Component        | Tools Used                                  |
|------------------|---------------------------------------------|
| UI & Dashboard   | Streamlit, HTML/CSS Markdown styling        |
| Model Training   | Scikit-learn (Random Forest)                |
| Data Handling    | Pandas                                      |
| Visualization    | Plotly                                      |
| Explainability   | SHAP (Summary Plot, Feature Importance)     |
| Deployment       | Streamlit Cloud                             |
| Branding         | Custom banner, dark red theme, QR-ready     |

---

## 🚀 Live Demo

[![Launch App](https://img.shields.io/badge/Launch%20App-FlightPulse-darkred?logo=streamlit)](https://share.streamlit.io/vikrantthenge/flight-delay-predictor/main/app.py)  
![Model](https://img.shields.io/badge/Model-Random%20Forest-green)  
![Explainability](https://img.shields.io/badge/Explainability-SHAP-blue)  
![Deployment](https://img.shields.io/badge/Deployed%20On-Streamlit%20Cloud-purple)  
[![CI Status](https://github.com/Vikrantthenge/flight-delay-predictor/actions/workflows/flightdelay-ci.yml/badge.svg)](https://github.com/Vikrantthenge/flight-delay-predictor/actions/workflows/flightdelay-ci.yml)

---

## 🔄 CI/CD Integration

This dashboard uses **GitHub Actions** for Continuous Integration and Deployment:

- ✅ Linting via `flake8` to ensure clean, error-free code  
- ✅ Optional unit testing via `pytest`  
- ✅ Auto-deployment to Streamlit Cloud on every push to `main`  
- ✅ Live CI badge reflects build health and workflow status  

Every update is automatically validated and deployed, ensuring production-grade reliability and faster iteration.

---

## 🧮 Model Explainability with SHAP

To enhance transparency and recruiter-facing clarity, FlightPulse integrates **SHAP (SHapley Additive exPlanations)** for model interpretability:

- 📌 **SHAP Summary Plot**: Visualizes global feature importance across all predictions  
- 🔍 **Top Feature Toggle**: Allows users to isolate and inspect the most influential variables  
- 🧠 **Model-Aware Insights**: Aligns SHAP values with `model.feature_names_in_` for schema consistency  
- 🎯 **Recruiter-Polished Output**: Ensures unclipped visuals and branded layout for professional presentation  

This layer of explainability helps users and recruiters understand why certain flight scenarios are predicted to be delayed, reinforcing trust and model transparency.

---

### 🔍 What Makes FlightPulse Unique

- ✈️ **Aviation + Weather Logic**  
  Inputs include departure hour, visibility, humidity, cloud cover, airline, origin, and destination — engineered to reflect real-world flight delay factors.

- 📊 **Branded Visual Experience**  
  Custom banner, multi-shade dark red palette, and recruiter-polished layout designed for visual impact and professional appeal.

- 🧠 **Model-Aware Input Engineering**  
  Dynamically aligns user inputs with trained feature schema using `model.feature_names_in_`, ensuring robust prediction flow.

- 🧮 **SHAP-Driven Explainability Layer**  
  Integrated SHAP plots and toggles provide recruiter-facing clarity on model logic and feature impact.

- 📱 **Mobile-Ready Deployment**  
  Hosted on Streamlit Cloud with responsive layout and QR-ready thumbnail for instant access from resume or LinkedIn.

- 🔗 **Live App + GitHub Integration**  
  Fully deployed with clean README, launch badge, and direct access to source code and model logic.
