import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def make_synthetic(n=1200, seed=7):
    rng = np.random.default_rng(seed)
    base_date = datetime(2022,1,1)
    rows = []
    for i in range(n):
        d = base_date + timedelta(days=i)
        t = i / 365 * 2 * math.pi
        temp = 18 + 8*math.sin(t) + rng.normal(0, 2)
        humidity = 55 + 20*math.sin(t + 0.7) + rng.normal(0, 5)
        wind = abs(3 + 2*math.sin(t + 1.2) + rng.normal(0, 1))
        pressure = 1010 + 6*math.sin(t + 2.5) + rng.normal(0, 2)
        traffic = np.clip(60 + 25*rng.normal() + 10*math.sin(2*t), 10, 100)
        holiday = 1 if d.weekday() >= 5 else 0
        pm25 = (0.55*traffic + 0.15*humidity - 3.5*wind - 0.05*pressure
                + rng.normal(0, 8) + (8 if holiday==0 and traffic>70 else -5))
        pm25 = float(np.clip(pm25, 5, 180))
        rows.append([d.date(), round(temp,2), round(humidity,1), round(wind,2), round(pressure,1),
                     round(traffic,1), holiday, round(pm25,1)])
    df = pd.DataFrame(rows, columns=[
        "date","temp_c","humidity","wind_speed","pressure_hpa",
        "traffic_index","holiday","pm25_next_day"
    ])
    return df

def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def plot_timeseries(df: pd.DataFrame, out_path: str | Path):
    ensure_dir(Path(out_path).parent)
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(df["date"], df["pm25_next_day"])
    ax.set_title("PM2.5 next-day (time series)")
    ax.set_ylabel("µg/m³"); ax.set_xlabel("date")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def plot_pred_vs_actual(y_true, y_pred, out_path: str | Path):
    ensure_dir(Path(out_path).parent)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(y_true, y_pred, alpha=0.6)
    mn, mx = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax.plot([mn, mx], [mn, mx], 'k--')
    ax.set_xlabel("Actual PM2.5"); ax.set_ylabel("Predicted PM2.5")
    ax.set_title("Predicted vs Actual")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
