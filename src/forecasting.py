from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# =========================
# 1) Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "outputs" / "daily_energy.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


# =========================
# 2) Load data
# =========================
print("Loading daily energy dataset...")
df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])


# =========================
# 3) Sort data
# =========================
df = df.sort_values(by=["id", "date"])


# =========================
# 4) Create lag features
# =========================
df["lag_1"] = df.groupby("id")["daily_energy_kwh"].shift(1)
df["lag_2"] = df.groupby("id")["daily_energy_kwh"].shift(2)
df["rolling_mean_3"] = df.groupby("id")["daily_energy_kwh"].shift(1).rolling(3).mean().reset_index(level=0, drop=True)


# =========================
# 5) Remove NaN (from lag)
# =========================
df = df.dropna()

print("\nDataset after lag creation:")
print(df.head())


# =========================
# 6) Prepare X and y
# =========================
X = df[["lag_1", "lag_2", "rolling_mean_3"]]
y = df["daily_energy_kwh"]


# =========================
# 7) Train/Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 8) Train model
# =========================
model = LinearRegression()
model.fit(X_train, y_train)


# =========================
# 9) Predictions
# =========================
y_pred = model.predict(X_test)


# =========================
# 10) Evaluation
# =========================
mae = mean_absolute_error(y_test, y_pred)

print("\nMean Absolute Error:", mae)


# =========================
# 11) Plot predictions vs real
# =========================
plt.figure(figsize=(8, 4))
plt.scatter(y_test, y_pred)
plt.xlabel("Real values")
plt.ylabel("Predictions")
plt.title("Forecasting: Real vs Predicted")

plot_path = OUTPUT_DIR / "forecasting_scatter.png"
plt.savefig(plot_path)
plt.close()

print(f"\nSaved plot: {plot_path}")