# app_streamlit.py
# 🌿 GreenPulse — Streamlit demo for SDG 13 (Climate Action)
# Author: Anne Atieno

# ------------------------------------------------------------
#  IMPORTS
# ------------------------------------------------------------
from pathlib import Path
import json
import pandas as pd
import joblib
import streamlit as st


# ------------------------------------------------------------
#  CONFIGURATION
# ------------------------------------------------------------
MODEL_PATH = Path("outputs/model.joblib")
METRICS_PATH = Path("outputs/metrics.json")

FEATURES = [
    "temp_c",
    "humidity",
    "wind_speed",
    "pressure_hpa",
    "traffic_index",
    "holiday",
]

# Streamlit page setup
st.set_page_config(
    page_title="GreenPulse • PM2.5 Forecast",
    page_icon="🌿",
    layout="centered"
)


# ------------------------------------------------------------
#  HELPER FUNCTIONS
# ------------------------------------------------------------
def load_model():
    """Load trained model from outputs folder."""
    if not MODEL_PATH.exists():
        return None, "Model file not found. Run: python src/train.py"
    try:
        model = joblib.load(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {e}"


def load_metrics():
    """Read metrics.json file."""
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text())
    except Exception:
        return None


def show_metric(label: str, value):
    """Display a metric safely."""
    if isinstance(value, (int, float)):
        st.metric(label, f"{value:.2f}")
    else:
        st.metric(label, "—")


# ------------------------------------------------------------
#  HEADER
# ------------------------------------------------------------
st.title("🌿 GreenPulse")
st.subheader("Forecast Tomorrow’s PM2.5 with Environmental and Activity Data")
st.markdown(
    "This prototype demonstrates how **AI can support SDG 13 (Climate Action)** "
    "by predicting next-day air quality (PM2.5) using simple inputs such as temperature, humidity, and traffic index."
)
st.caption("Prototype by Anne Atieno • AI for Sustainable Development")


# ------------------------------------------------------------
#  SIDEBAR INPUTS
# ------------------------------------------------------------
st.sidebar.header("Adjust Input Values")

temp_c = st.sidebar.slider("Temperature (°C)", 5.0, 45.0, 26.0, 0.1)
humidity = st.sidebar.slider("Humidity (%)", 5.0, 100.0, 58.0, 0.1)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 20.0, 3.2, 0.1)
pressure_hpa = st.sidebar.slider("Pressure (hPa)", 980.0, 1040.0, 1011.0, 0.1)
traffic_index = st.sidebar.slider("Traffic Index (0–100)", 0.0, 100.0, 72.0, 1.0)
holiday = st.sidebar.checkbox("Is Holiday?", value=False)


# ------------------------------------------------------------
#  LOAD MODEL + METRICS
# ------------------------------------------------------------
model, model_error = load_model()
metrics = load_metrics()


# ------------------------------------------------------------
#  METRICS DISPLAY
# ------------------------------------------------------------
st.markdown("### 📊 Model Evaluation Results")
c1, c2, c3 = st.columns(3)
if metrics:
    with c1:
        show_metric("MAE (µg/m³)", metrics.get("MAE"))
    with c2:
        show_metric("RMSE (µg/m³)", metrics.get("RMSE"))
    with c3:
        show_metric("R² Score", metrics.get("R2"))
else:
    st.info("No metrics yet. Please run `python src/train.py` to train the model.")


# ------------------------------------------------------------
#  PREDICTION SECTION
# ------------------------------------------------------------
st.markdown("### 🔮 Predict Next-Day PM2.5")
if model_error:
    st.warning(model_error)
elif model is None:
    st.warning("Model not available.")
else:
    # Prepare DataFrame
    X = pd.DataFrame([{
        "temp_c": float(temp_c),
        "humidity": float(humidity),
        "wind_speed": float(wind_speed),
        "pressure_hpa": float(pressure_hpa),
        "traffic_index": float(traffic_index),
        "holiday": int(holiday),
    }], columns=FEATURES)

    try:
        pred = float(model.predict(X)[0])
        st.success(f"**Predicted next-day PM2.5: {pred:.2f} µg/m³**")

        # Qualitative interpretation
        if pred < 25:
            st.caption("🟢 Good Air Quality")
        elif pred < 50:
            st.caption("🟡 Moderate — Sensitive groups should take care")
        elif pred < 100:
            st.caption("🟠 Unhealthy for Sensitive Groups — Consider advisories")
        else:
            st.caption("🔴 Unhealthy — Avoid outdoor activity where possible")

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ------------------------------------------------------------
#  VISUAL OUTPUTS
# ------------------------------------------------------------
st.markdown("### 📈 Training Visualisations")

pva = Path("outputs/pred_vs_actual.png")
ts = Path("outputs/pm25_timeseries.png")
sh = Path("outputs/shap_beeswarm.png")

if pva.exists():
    st.image(str(pva), caption="Predicted vs Actual PM2.5", use_container_width=True)
if ts.exists():
    st.image(str(ts), caption="PM2.5 Time Series", use_container_width=True)
if sh.exists():
    st.image(str(sh), caption="Feature Importance (SHAP)", use_container_width=True)


# ------------------------------------------------------------
#  FOOTER
# ------------------------------------------------------------
st.divider()
st.caption("© 2025 GreenPulse • Prototype by Anne Atieno • AI for Sustainable Development (SDG 13)")
