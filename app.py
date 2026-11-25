from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from model import cfg
from utils import load_dataset, fit_notebook_scaler, load_model, forecast, metrics_for_sensor

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "METR-LA.h5"
MODEL_PATH = ROOT / "checkpoints" / "HGFNet.pth"

st.set_page_config(page_title="HGFNet | Traffic Forecasting", page_icon="🚦", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
.hero {padding: 1.25rem 1.5rem; border-radius: 16px; background: linear-gradient(110deg,#10233f,#174b68); color:white; margin-bottom:1rem;}
.hero h1 {color:white; margin:0; font-size:2.1rem;}
.hero p {color:#d5e5f4; margin:.35rem 0 0 0;}
.metric-card {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:14px;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading METR-LA dataset...")
def get_data(path, mtime):
    return load_dataset(path)

@st.cache_resource(show_spinner="Loading trained HGFNet checkpoint...")
def get_model(path, mtime, device_name):
    return load_model(path, torch.device(device_name))

st.markdown('<div class="hero"><h1>🚦 HGFNet</h1><p>Hypergraph and Frequency-Aware Traffic Speed Forecasting</p></div>', unsafe_allow_html=True)

if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.error("Required files are missing. Keep METR-LA.h5 in data/ and HGFNet.pth in checkpoints/.")
    st.stop()

device_name = "cuda" if torch.cuda.is_available() else "cpu"
try:
    df, raw = get_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    scaler, _ = fit_notebook_scaler(raw)
    model = get_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime, device_name)
except Exception as exc:
    st.exception(exc)
    st.stop()

n_time, n_sensors = raw.shape
st.caption(f"Checkpoint loaded successfully · Device: {device_name.upper()} · Dataset: METR-LA")

m1,m2,m3,m4 = st.columns(4)
m1.metric("Time steps", f"{n_time:,}")
m2.metric("Traffic sensors", f"{n_sensors}")
m3.metric("Sampling interval", "5 minutes")
m4.metric("Forecast horizon", "60 minutes")

with st.sidebar:
    st.header("Forecast controls")
    sensor = st.selectbox("Traffic sensor", options=list(range(n_sensors)), index=0)
    end_index = st.slider("Last observed time-step index", min_value=cfg.history-1, max_value=n_time-1, value=n_time-cfg.horizon-1)
    st.caption("The model uses the 12 observations ending at the selected index.")
    run = st.button("Generate forecast", type="primary", use_container_width=True)
    st.divider()
    st.subheader("Model configuration")
    st.write("Input window: **12 steps (60 min)**")
    st.write("Forecast: **12 steps (60 min)**")
    st.write("Embedding: **64**")
    st.write("Hyperedges: **64**")
    st.write("Attention heads: **8**")

# Predict on initial page load as well; button reruns and refreshes.
try:
    hist, pred_all, hist_start = forecast(model, raw, scaler, end_index, torch.device(device_name))
except Exception as exc:
    st.error(f"Forecast failed: {exc}")
    st.stop()

sensor_actual_history = hist[:, sensor]
sensor_pred = pred_all[:, sensor]
future_index = np.arange(end_index+1, end_index+1+cfg.horizon)
# Use source dataframe index when possible; otherwise use integer step labels.
try:
    hist_labels = pd.to_datetime(df.index[hist_start:end_index+1])
    future_labels = pd.date_range(hist_labels[-1] + pd.Timedelta(minutes=5), periods=cfg.horizon, freq="5min")
    x_hist = hist_labels
    x_future = future_labels
    x_title = "Time"
except Exception:
    x_hist = np.arange(hist_start,end_index+1)
    x_future = future_index
    x_title = "Time-step index"

left,right = st.columns([1.7,1])
with left:
    st.subheader(f"Traffic speed forecast · Sensor {sensor}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_hist, y=sensor_actual_history, mode="lines+markers", name="Historical speed", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=x_future, y=sensor_pred, mode="lines+markers", name="Predicted speed", line=dict(width=3, dash="dash")))
    fig.add_vline(x=x_hist[-1], line_dash="dot", line_color="gray", annotation_text="Forecast starts")
    fig.update_layout(height=430, xaxis_title=x_title, yaxis_title="Speed (mph)", hovermode="x unified", legend=dict(orientation="h", y=1.12), margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Predicted speeds")
    forecast_table = pd.DataFrame({
        "Horizon": [f"{(i+1)*5} min" for i in range(cfg.horizon)],
        "Predicted speed (mph)": np.round(sensor_pred, 2)
    })
    st.dataframe(forecast_table, use_container_width=True, hide_index=True, height=430)

st.subheader("Forecast across all sensors")
all_cols = st.columns(3)
for j, horizon_min in enumerate([15,30,60]):
    step = horizon_min//5 - 1
    vals = pred_all[step]
    all_cols[j].metric(f"Forecast at {horizon_min} min · Sensor {sensor}", f"{sensor_pred[step]:.2f} mph")

with st.expander("View predictions for all 207 sensors"):
    all_df = pd.DataFrame(pred_all, columns=[f"Sensor {i}" for i in range(n_sensors)])
    all_df.insert(0, "Forecast step", [f"{(i+1)*5} min" for i in range(cfg.horizon)])
    st.dataframe(all_df, use_container_width=True)

st.subheader("Forecasting performance")
st.info("This live interface generates predictions from the selected historical window. Ground-truth future values are not used for the forecast. Evaluation metrics require a test window with known future observations.")
if end_index + cfg.horizon < n_time:
    actual_future = raw[end_index+1:end_index+1+cfg.horizon, sensor]
    score = metrics_for_sensor(actual_future, sensor_pred)
    c1,c2,c3 = st.columns(3)
    c1.metric("MAE", f"{score['MAE']:.3f} mph")
    c2.metric("RMSE", f"{score['RMSE']:.3f} mph")
    c3.metric("MAPE", f"{score['MAPE']:.2f}%")
    st.caption("These are retrospective metrics for this selected window; zero/missing actual readings are excluded. They are not the report's aggregate test-set metrics.")
else:
    st.warning("The selected index is too close to the end of the dataset to calculate retrospective metrics.")

with st.expander("About this implementation"):
    st.write("""
    This application uses the original HGFNet architecture and checkpoint. The notebook's
    StandardScaler procedure is recreated by fitting on the first 70% of the METR-LA time
    series, independently for each sensor. The model consumes 12 historical steps and
    predicts 12 future steps. The notebook's temporal block is implemented as gated
    depthwise Conv1D; it is not a dedicated state-space model.
    """)
    st.write("The checkpoint contains model weights only. The scaler is reconstructed from the included dataset.")
