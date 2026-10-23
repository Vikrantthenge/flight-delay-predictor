
import io
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import qrcode
import shap
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Flight Pulse | Delay Risk Predictor",
    page_icon="✈️",
    layout="wide",
)

CUSTOM_REDS = ["#4A0000", "#800000", "#8B0000", "#B22222", "#DC143C"]
MODEL_PATH = Path(__file__).parent / "model" / "flight_delay_model.pkl"

AIRLINE_FALLBACK = [
    "Air India",
    "Akasa Air",
    "Indigo",
    "Spicejet",
    "Vistara",
    "Air Asia",
]

AIRPORT_FALLBACK = ["BOM", "CCU", "DEL", "HYD", "MAA", "TRV"]


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


st.markdown(
    """
    <div style="background:#8B0000;padding:8px;text-align:center;
                border-radius:5px;color:white;">
        FLIGHT PULSE
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="text-align:center;padding:12px 0;">
        <h1 style="color:darkred;">✈️ Flight Pulse</h1>
        <h3 style="color:gray;">Flight Delay Risk Prediction Dashboard</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: {MODEL_PATH}. "
        "Ensure model/flight_delay_model.pkl exists in your repository."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Unable to load the saved model: {exc}")
    st.stop()


# ============================================================
# MODEL FEATURE DEFINITIONS
# ============================================================

MODEL_FEATURES = list(
    getattr(
        model,
        "feature_names_in_",
        [
            "Dep_Hour",
            "Arr_Hour",
            "weather__hourly__visibility",
            "weather__hourly__humidity",
            "weather__hourly__cloudcover",
        ],
    )
)


def categories_from_features(prefix, fallback):
    """Use categories represented in the saved model where possible."""
    found = [
        name[len(prefix):]
        for name in MODEL_FEATURES
        if name.startswith(prefix)
    ]
    return found if found else fallback


AIRLINES = categories_from_features("Airline_", AIRLINE_FALLBACK)
ORIGINS = categories_from_features("From_", AIRPORT_FALLBACK)
DESTINATIONS = categories_from_features("TO_", AIRPORT_FALLBACK)


def create_input_df(
    dep_hour,
    arr_hour,
    visibility,
    humidity,
    cloudcover,
    airline,
    origin,
    destination,
):
    """Create an input row using the saved model's feature order."""

    input_dict = {
        "Dep_Hour": dep_hour,
        "Arr_Hour": arr_hour,
        "weather__hourly__visibility": visibility,
        "weather__hourly__humidity": humidity,
        "weather__hourly__cloudcover": cloudcover,
    }

    for name in MODEL_FEATURES:
        if name.startswith("Airline_"):
            category = name[len("Airline_"):]
            input_dict[name] = int(airline == category)

        elif name.startswith("From_"):
            category = name[len("From_"):]
            input_dict[name] = int(origin == category)

        elif name.startswith("TO_"):
            category = name[len("TO_"):]
            input_dict[name] = int(destination == category)

    # Missing model features are filled with zero, as in the
    # original application. Confirm this matches training preprocessing.
    input_df = pd.DataFrame([input_dict])

    for column in MODEL_FEATURES:
        if column not in input_df.columns:
            input_df[column] = 0

    return input_df[MODEL_FEATURES]


def get_model_name(estimator):
    """Identify the loaded estimator rather than guessing its algorithm."""
    name = type(estimator).__name__

    if name == "RandomForestClassifier":
        trees = getattr(estimator, "n_estimators", None)
        return f"Random Forest ({trees} trees)" if trees else name

    return name


def get_positive_class_probability(estimator, input_df):
    """Return the probability for the delayed-flight class."""
    if not hasattr(estimator, "predict_proba"):
        raise ValueError(
            "The loaded model does not support predict_proba()."
        )

    probabilities = estimator.predict_proba(input_df)[0]
    classes = list(getattr(estimator, "classes_", []))

    if 1 in classes:
        positive_index = classes.index(1)
    elif True in classes:
        positive_index = classes.index(True)
    else:
        raise ValueError(
            "Could not identify class 1 as the delayed-flight class. "
            "Check the target labels used during model training."
        )

    return float(probabilities[positive_index])


# ============================================================
# OVERVIEW
# ============================================================

st.markdown("## 🧭 Project Overview")

st.markdown(
    """
    Flight Pulse is an interactive application for exploring flight
    delay risk using schedule, airline, route, and weather inputs.

    The application loads a previously trained machine learning model
    and provides prediction probabilities, model feature importance
    where supported, and SHAP-based explanations.

    **Scope:** This is a demonstration application. It does not connect
    to live flight schedules, airline operations systems, or live weather
    feeds.
    """
)

with st.expander("📘 Model and evaluation information"):
    st.markdown(
        f"""
        - **Loaded estimator:** `{get_model_name(model)}`
        - **Prediction output:** Model-estimated class probability
        - **Reported test accuracy:** Approximately 72%, pending
          verification against the training and evaluation script
        - **Reported baseline:** 67.5%, pending verification
        - **Explainability:** SHAP, where supported by the estimator
        """
    )
    st.caption(
        "The accuracy and baseline above are reported figures, not "
        "metrics recalculated by this application."
    )


# ============================================================
# SIDEBAR AND INPUTS
# ============================================================

with st.sidebar:
    st.markdown("✈️ **Flight Pulse**")

    qr_url = "https://flight-delay-predictor-pulse.streamlit.app/"

    try:
        qr_image = qrcode.make(qr_url)
        st.image(qr_image, caption="Scan to open Flight Pulse", width=140)
    except Exception:
        st.markdown(f"[Open Flight Pulse]({qr_url})")

    st.markdown("### 🔍 Model Information")
    st.write(f"**Loaded model:** {get_model_name(model)}")
    st.write("**Data source:** Historical model inputs")
    st.write("**Mode:** Interactive demonstration")

    st.divider()
    st.header("Flight Details")

    dep_hour = st.slider("Scheduled Departure Hour", 0, 23, 9)
    arr_hour = st.slider("Scheduled Arrival Hour", 0, 23, 11)

    visibility = st.slider("Visibility (km)", 1, 10, 5)
    humidity = st.slider("Humidity (%)", 10, 100, 60)
    cloudcover = st.slider("Cloud Cover (%)", 0, 100, 40)

    airline = st.selectbox("Airline", AIRLINES)
    origin = st.selectbox("Origin Airport", ORIGINS)
    destination = st.selectbox("Destination Airport", DESTINATIONS)

    predict_clicked = st.button(
        "Predict Delay Risk",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# PREDICTION
# ============================================================

st.markdown("## ✈️ Delay Risk Assessment")

if predict_clicked:
    try:
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

        prediction = get_positive_class_probability(model, input_df)

        st.metric(
            label="Model-Estimated Delay Probability",
            value=f"{prediction * 100:.1f}%",
        )

        st.caption(
            "This is the model's estimated probability for the selected "
            "inputs. It is not a guaranteed outcome or a live flight forecast."
        )

        with st.expander("Inspect model inputs"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as exc:
        st.error(f"Prediction could not be generated: {exc}")
else:
    st.info(
        "Choose the flight and weather inputs in the sidebar, "
        "then select **Predict Delay Risk**."
    )


# ============================================================
# ILLUSTRATIVE CHARTS
# ============================================================

st.markdown("---")
st.markdown("## 📊 Illustrative Operations Charts")

st.warning(
    "The following charts use fixed example values from the original "
    "application. They are illustrative only and are not calculated "
    "from the training dataset or live airline operations data."
)

airline_colors = {
    "Indigo": "#3F51B5",
    "Akasa Air": "#FFA500",
    "Air India": "#B22222",
    "Vistara": "#4B2E2E",
    "Spicejet": "#800000",
    "Air Asia": "#A52A2A",
}

airline_delay_df = pd.DataFrame(
    {
        "Airline": [
            "Indigo",
            "Spicejet",
            "Air India",
            "Akasa Air",
            "Air Asia",
            "Vistara",
        ],
        "Avg Delay (min)": [12, 18, 22, 15, 17, 9],
    }
)

st.subheader("Average Delay by Airline: Example Values")

fig_bar = px.bar(
    airline_delay_df,
    x="Airline",
    y="Avg Delay (min)",
    color="Airline",
    color_discrete_map=airline_colors,
    title="Illustrative Average Delay by Airline",
    labels={"Avg Delay (min)": "Example delay (minutes)"},
)

st.plotly_chart(fig_bar, use_container_width=True)


hourly_delay_df = pd.DataFrame(
    {
        "Hour": list(range(24)),
        "Avg Delay (min)": [
            5, 6, 8, 10, 12, 15, 18, 20,
            22, 19, 17, 14, 12, 10, 9, 8,
            7, 6, 5, 4, 4, 5, 6, 7,
        ],
    }
)

st.subheader("Delay by Departure Hour: Example Values")

fig_line = px.line(
    hourly_delay_df,
    x="Hour",
    y="Avg Delay (min)",
    markers=True,
    title="Illustrative Delay Pattern by Departure Hour",
    labels={
        "Hour": "Departure hour",
        "Avg Delay (min)": "Example delay (minutes)",
    },
)

st.plotly_chart(fig_line, use_container_width=True)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.markdown("---")
st.markdown("## 📌 Model Feature Importance")

if hasattr(model, "feature_importances_"):
    importance_df = pd.DataFrame(
        {
            "Feature": MODEL_FEATURES,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    fig_importance = px.bar(
        importance_df.head(10).sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Model Feature Importances",
        color_discrete_sequence=[CUSTOM_REDS[3]],
    )

    st.plotly_chart(fig_importance, use_container_width=True)

    st.caption(
        "Feature importance measures model reliance under the estimator's "
        "importance definition. It does not establish causation."
    )
else:
    st.info(
        "The loaded estimator does not expose feature_importances_. "
        "Use an explanation method appropriate for this model."
    )


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

st.markdown("---")
st.markdown("## 🧠 SHAP Explainability")

st.markdown(
    """
    SHAP estimates how input features contribute to model predictions.
    The summary below uses a small set of example scenarios. It is not
    an explanation of the current prediction and should not be interpreted
    as proof that a feature causes flight delays.
    """
)

try:
    sample_inputs = pd.concat(
        [
            create_input_df(
                9, 11, 5, 60, 40, AIRLINES[0],
                ORIGINS[0], DESTINATIONS[0],
            ),
            create_input_df(
                14, 16, 8, 70, 20, AIRLINES[0],
                ORIGINS[0], DESTINATIONS[0],
            ),
            create_input_df(
                6, 8, 3, 50, 60, AIRLINES[0],
                ORIGINS[0], DESTINATIONS[0],
            ),
            create_input_df(
                18, 20, 9, 85, 10, AIRLINES[0],
                ORIGINS[0], DESTINATIONS[0],
            ),
            create_input_df(
                22, 1, 2, 40, 90, AIRLINES[0],
                ORIGINS[0], DESTINATIONS[0],
            ),
        ],
        ignore_index=True,
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_inputs)

    # Handle common SHAP output formats for binary classifiers.
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            plot_values = shap_values[1]
        else:
            plot_values = shap_values[0]
    else:
        plot_values = shap_values

        if getattr(plot_values, "ndim", 0) == 3:
            # Common shape: samples x features x classes.
            plot_values = plot_values[:, :, -1]

    fig, ax = plt.subplots(figsize=(10, 6))

    shap.summary_plot(
        plot_values,
        sample_inputs,
        plot_type="bar",
        show=False,
        max_display=10,
    )

    fig = plt.gcf()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    buffer.seek(0)
    plt.close(fig)

    st.image(buffer, caption="SHAP importance across example scenarios")

    with st.expander("How to interpret this chart"):
        st.markdown(
            """
            - Longer bars indicate greater average absolute SHAP impact
              across the supplied example scenarios.
            - This bar chart shows impact magnitude, not whether a feature
              increases or decreases delay risk.
            - It uses only five example rows, so the ranking may be unstable.
            - SHAP explains the model's behavior, not the true causes of delays.
            """
        )

except Exception as exc:
    st.warning(
        "SHAP could not generate the explanation for this model. "
        f"Technical detail: {exc}"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
        <p><strong style="color:darkred;">
            Flight Pulse | Flight Delay Risk Demonstration
        </strong></p>
        <p>
            <a href="https://github.com/vikrantthenge" target="_blank">
                GitHub
            </a>
            &nbsp; | &nbsp;
            <a href="https://linkedin.com/in/vthenge" target="_blank">
                LinkedIn
            </a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
