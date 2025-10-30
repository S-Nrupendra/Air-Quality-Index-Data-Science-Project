import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------------
# 🌤️ PAGE CONFIGURATION
# ---------------------------
st.set_page_config(
    page_title="AQI Prediction App",
    page_icon="🌫️",
    layout="centered",
)

# ---------------------------
# 🧠 UTILITY FUNCTIONS
# ---------------------------

def load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Failed to load file: {path}\nError: {e}")
        return None

def assign_aqi_bucket(aqi):
    """Assign AQI category based on value"""
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
# 🚀 LOAD MODELS AND SCALERS
# ---------------------------

model_with = load_pickle("best_model_with_Xylene.pkl")
model_without = load_pickle("best_model.pkl")

scaler_with = load_pickle("scaler_with.pkl")
scaler_without = load_pickle("scaler_without.pkl")

if not all([model_with, model_without, scaler_with, scaler_without]):
    st.error("❌ Missing model or scaler file(s). Please ensure all are present.")
    st.stop()

# ---------------------------
# 🧩 FEATURE DEFINITIONS
# ---------------------------

features_without = [
    'PM2.5', 'PM10', 'NO', 'NO2', 'NOx',
    'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene'
]

features_with = features_without + ['Xylene']

units = {
    'PM2.5': 'µg/m³',
    'PM10': 'µg/m³',
    'NO': 'µg/m³',
    'NO2': 'µg/m³',
    'NOx': 'µg/m³',
    'NH3': 'µg/m³',
    'CO': 'mg/m³',
    'SO2': 'µg/m³',
    'O3': 'µg/m³',
    'Benzene': 'µg/m³',
    'Toluene': 'µg/m³',
    'Xylene': 'µg/m³'
}

# ---------------------------
# 🎛️ APP INTERFACE
# ---------------------------

st.title("🌫️ Air Quality Index (AQI) Prediction")
st.markdown("""
Enter pollutant concentrations below to predict the **Air Quality Index (AQI)** and its corresponding category.
""")

model_choice = st.radio(
    "Choose Model Type:",
    ["With Xylene", "Without Xylene"],
    horizontal=True
)

st.markdown("### 🧪 Enter Pollutant Levels")

# Input fields layout
cols = st.columns(3)
user_input = {}

selected_features = features_with if model_choice == "With Xylene" else features_without

for i, feature in enumerate(selected_features):
    with cols[i % 3]:
        user_input[feature] = st.number_input(
            f"{feature} ({units[feature]})",
            min_value=0.0,
            step=0.01,
            format="%.2f"
        )

# ---------------------------
# 🔮 PREDICTION
# ---------------------------

if st.button("🔍 Predict AQI"):
    input_df = pd.DataFrame([user_input])

    # Select correct model and scaler
    if model_choice == "With Xylene":
        model = model_with
        scaler = scaler_with
    else:
        model = model_without
        scaler = scaler_without

    # Scale input (same as during training)
    try:
        input_scaled = scaler.transform(input_df.values)
        predicted_aqi = model.predict(input_scaled)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # ---------------------------
    # 🧭 CALIBRATION & CLIPPING
    # ---------------------------
    # Optional simple calibration — shift/scale predicted range to realistic AQI domain
    # (You can tune these coefficients using validation data)
    calibrated_aqi = 1.05 * predicted_aqi + 10  # mild upward correction

    # Clip to valid AQI range (0–500)
    calibrated_aqi = np.clip(calibrated_aqi, 0, 500)

    # Assign AQI bucket
    bucket, emoji = assign_aqi_bucket(calibrated_aqi)

    # ---------------------------
    # 🌈 RESULTS DISPLAY
    # ---------------------------
    st.markdown("---")
    st.markdown("## 🌡️ Predicted Results")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted AQI", value=f"{calibrated_aqi:.2f}")
    with col2:
        st.markdown(f"### {emoji} Air Quality Category: **{bucket}**")

    # Optional guidance
    st.info("""
**AQI Ranges:**
- 0–50: Good  
- 51–100: Satisfactory  
- 101–200: Moderate  
- 201–300: Poor  
- 301–400: Very Poor  
- 401–500: Severe
""")

# ---------------------------
# 📘 FOOTER
# ---------------------------
st.markdown("---")
st.caption("Developed by [Your Name] | Powered by Streamlit & XGBoost 🌿")
