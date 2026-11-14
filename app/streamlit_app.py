
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Household Energy Prediction",
    page_icon="⚡",
    layout="wide"
)

# Load saved model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "energy_pipeline.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("Trained model file not found.")
    st.stop()

saved_model = load_model()
pipeline = saved_model["pipeline"]
features = saved_model["features"]

# Main heading
st.title("⚡ Household Energy Consumption Prediction")

st.write(
    "Predict hourly household energy consumption "
    "using temperature, humidity and weather conditions."
)

st.info(
    "ML Model: Ridge Regression | PCA: 11 components | "
    "Test R²: 0.14"
)

st.caption(
    "Default values are examples. Replace them with actual "
    "sensor and weather measurements."
)

# Input form
with st.form("prediction_form"):

    st.header("🏠 Indoor Sensor Measurements")
    st.write("Enter temperature and humidity for all 9 sensor locations.")

    user_inputs = {}

    columns = st.columns(3)

    for i in range(1, 10):
        with columns[(i - 1) % 3]:

            st.subheader(f"Sensor {i}")

            user_inputs[f"T{i}"] = st.number_input(
                f"Temperature T{i} (°C)",
                value=20.0,
                step=0.1,
                key=f"T{i}"
            )

            user_inputs[f"RH_{i}"] = st.number_input(
                f"Humidity RH_{i} (%)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=0.1,
                key=f"RH_{i}"
            )

    st.divider()

    st.header("🌤️ Outdoor Weather Measurements")

    outdoor_columns = st.columns(3)

    with outdoor_columns[0]:
        user_inputs["T_out"] = st.number_input(
            "Outdoor Temperature (°C)", value=10.0
        )
        user_inputs["RH_out"] = st.number_input(
            "Outdoor Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0
        )

    with outdoor_columns[1]:
        user_inputs["Press_mm_hg"] = st.number_input(
            "Atmospheric Pressure (mmHg)", value=760.0
        )
        user_inputs["Windspeed"] = st.number_input(
            "Wind Speed (dataset units)",
            min_value=0.0,
            value=2.0
        )

    with outdoor_columns[2]:
        user_inputs["Visibility"] = st.number_input(
            "Visibility (dataset units)",
            min_value=0.0,
            value=40.0
        )
        user_inputs["Tdewpoint"] = st.number_input(
            "Dew Point Temperature (°C)", value=4.0
        )

    submitted = st.form_submit_button(
        "⚡ Predict Energy Consumption",
        type="primary",
        use_container_width=True
    )

# Make prediction
if submitted:

    # Use exactly the feature order from training
    input_data = pd.DataFrame([
        {feature: user_inputs[feature] for feature in features}
    ])

    predicted_wh = float(pipeline.predict(input_data)[0])

    st.divider()
    st.header("📊 Prediction Result")

    if predicted_wh < 0:
        st.warning(
            "The model predicted negative energy consumption. "
            "This is physically invalid; review the inputs."
        )
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Hourly Energy",
                f"{predicted_wh:,.2f} Wh"
            )

        with col2:
            st.metric(
                "Energy in kWh",
                f"{predicted_wh / 1000:,.3f} kWh"
            )

    st.caption(
        "This is a machine learning estimate, not a direct "
        "electricity meter reading."
    )

st.divider()
st.caption("DS605 | MSc Data Science | Machine Learning Project")
