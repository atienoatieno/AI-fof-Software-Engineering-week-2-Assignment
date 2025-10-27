import argparse, joblib, pandas as pd
from pathlib import Path

MODEL_PATH = Path("outputs/model.joblib")

def main(a):
    if not MODEL_PATH.exists():
        raise SystemExit("⚠️ Model not found. Run: python src/train.py first.")
    pipe = joblib.load(MODEL_PATH)
    x = pd.DataFrame([{
        "temp_c": a.temp, "humidity": a.humidity, "wind_speed": a.wind,
        "pressure_hpa": a.pressure, "traffic_index": a.traffic, "holiday": a.holiday
    }])
    pred = float(pipe.predict(x)[0])
    print(f"Predicted next-day PM2.5: {pred:.2f} µg/m³")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--temp", type=float, required=True)
    p.add_argument("--humidity", type=float, required=True)
    p.add_argument("--wind", type=float, required=True)
    p.add_argument("--pressure", type=float, required=True)
    p.add_argument("--traffic", type=float, required=True)
    p.add_argument("--holiday", type=int, choices=[0,1], required=True)
    args = p.parse_args()
    main(args)
