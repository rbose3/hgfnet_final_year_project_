from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from model import HGFNet, cfg

def load_dataset(path):
    # Matches notebook: pd.read_hdf(path), then df.values
    df = pd.read_hdf(path)
    values = df.values.astype(np.float32)
    if values.shape[1] != cfg.num_nodes:
        raise ValueError(f"Expected 207 sensors, found {values.shape[1]}.")
    if values.shape[0] < cfg.history:
        raise ValueError("Dataset does not contain enough time steps.")
    return df, values

def fit_notebook_scaler(raw_values):
    train_size = int(0.7 * len(raw_values))
    scaler = StandardScaler()
    scaler.fit(raw_values[:train_size])
    scaled = scaler.transform(raw_values)
    return scaler, scaled

def load_model(checkpoint_path, device):
    model = HGFNet().to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

@torch.inference_mode()
def forecast(model, raw_values, scaler, end_index, device):
    start = end_index - cfg.history + 1
    if start < 0 or end_index >= len(raw_values):
        raise ValueError("Selected end index is outside the valid range.")
    scaled_all = scaler.transform(raw_values)
    x_scaled = scaled_all[start:end_index+1]
    x = torch.tensor(x_scaled, dtype=torch.float32, device=device)[None, :, :, None]
    pred_scaled = model(x).squeeze(-1).squeeze(0).cpu().numpy()  # (12,207)
    pred_raw = scaler.inverse_transform(pred_scaled)
    history_raw = raw_values[start:end_index+1]
    return history_raw, pred_raw, start

def metrics_for_sensor(actual, predicted):
    valid = np.isfinite(actual) & np.isfinite(predicted) & (actual > 0)
    if not valid.any():
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
    err = predicted[valid] - actual[valid]
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE": float(np.mean(np.abs(err) / np.maximum(np.abs(actual[valid]), 1e-6)) * 100),
    }
