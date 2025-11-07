import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# ---------------------------
# 🌤️ PAGE CONFIGURATION
# ---------------------------
st.set_page_config(page_title="AQI Prediction App", page_icon="🌫️", layout="centered")

# ---------------------------
# 🧠 UTILITY FUNCTIONS
# ---------------------------

def load_xgb_model(path):
    try:
        model = XGBRegressor()
        model.load_model(path)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model from {path}: {e}")
        return None

def assign_aqi_bucket(aqi):
    if aqi <= 50:
        return "Good", "🟢"
    elif aqi <= 100:
        return "Satisfactory", "🟡"
    elif aqi <= 200:
        return "Moderate", "🟠"
    elif aqi <= 300:
        return "Poor", "🔴"
    elif aqi <= 400:
        return "Very Poor", "🟣"
    else:
        return "Severe", "⚫"

# ---------------------------
# 🚀 LOAD MODELS
# ---------------------------

model_with = load_xgb_model("best_model_With_Xylene.json")
model_without = load_xgb_model("best_model_Without_Xylene.json")

if not all([model_with, model_without]):
    st.error("❌ Missing or invalid JSON model files. Please ensure both are present.")
    st.stop()
else:
    st.success("✅ Models loaded successfully!")

# ---------------------------
# 🧩 FEATURE DEFINITIONS
# ---------------------------

features_without = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx',
                    'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']
features_with = features_without + ['Xylene']

units = {
    'PM2.5': 'µg/m³', 'PM10': 'µg/m³', 'NO': 'µg/m³', 'NO2': 'µg/m³',
    'NOx': 'µg/m³', 'NH3': 'µg/m³', 'CO': 'mg/m³', 'SO2': 'µg/m³',
    'O3': 'µg/m³', 'Benzene': 'µg/m³', 'Toluene': 'µg/m³', 'Xylene': 'µg/m³'
}

# ---------------------------
# 🎛️ APP INTERFACE
# ---------------------------
st.title("🌫️ Air Quality Index (AQI) Prediction")
st.markdown("Enter pollutant concentrations to predict **AQI** and its category.")

model_choice = st.radio("Choose Model Type:", ["With Xylene", "Without Xylene"], horizontal=True)

st.markdown("### 🧪 Enter Pollutant Levels")
cols = st.columns(3)
user_input = {}

selected_features = features_with if model_choice == "With Xylene" else features_without

for i, feature in enumerate(selected_features):
    with cols[i % 3]:
        user_input[feature] = st.number_input(f"{feature} ({units[feature]})", min_value=0.0, step=0.01, format="%.2f")

# ---------------------------
# 🔮 PREDICTION
# ---------------------------
if st.button("🔍 Predict AQI"):
    input_df = pd.DataFrame([user_input])
    expected_features = features_with if model_choice == "With Xylene" else features_without
    model = model_with if model_choice == "With Xylene" else model_without

    # Ensure column order consistency
    input_df = input_df[expected_features]

    # Debug panel
    with st.expander("🧠 Debug Info"):
        st.write("Input to model:")
        st.dataframe(input_df)
        try:
            st.write("Model features:", model.get_booster().feature_names)
        except Exception as e:
            st.write("Model features unavailable:", e)

    try:
        predicted_aqi = model.predict(input_df)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    predicted_aqi = np.clip(predicted_aqi, 0, 500)
    bucket, emoji = assign_aqi_bucket(predicted_aqi)

    st.markdown("---")
    st.markdown("## 🌡️ Predicted Air Quality")
    col1, col2 = st.columns(2)
    col1.metric(label="Predicted AQI", value=f"{predicted_aqi:.2f}")
    col2.markdown(f"### {emoji} Air Quality Category: **{bucket}**")
