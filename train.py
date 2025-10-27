
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib, json, matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# Dummy synthetic data for example
np.random.seed(0)
df = pd.DataFrame({
    "temperature": np.random.uniform(15, 35, 200),
    "humidity": np.random.uniform(30, 90, 200),
    "wind_speed": np.random.uniform(1, 10, 200),
    "pressure": np.random.uniform(1000, 1020, 200),
    "traffic": np.random.uniform(50, 100, 200),
    "holiday": np.random.randint(0, 2, 200),
})
df["pm25"] = (
    0.5 * df["temperature"]
    + 0.2 * df["humidity"]
    - 0.1 * df["wind_speed"]
    + 0.3 * df["traffic"]
    + np.random.normal(0, 5, 200)
)

X = df.drop("pm25", axis=1)
y = df["pm25"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
print(metrics)

with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
joblib.dump(model, OUT_DIR / "model.joblib")

plt.scatter(y_test, y_pred)
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Prediction vs Actual")
plt.savefig(OUT_DIR / "pred_vs_actual.png", dpi=160)
plt.close()

print("✅ Training complete. Results saved in 'outputs/'")
