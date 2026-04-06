from pathlib import Path
import pandas as pd
import numpy as np


# =========================
# 1) Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "outputs" / "sample_data.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "features.csv"


# =========================
# 2) Load data
# =========================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Convert datetime again
df["horodate"] = pd.to_datetime(df["horodate"], utc=True)
df["date"] = pd.to_datetime(df["date"])


# =========================
# 3) Basic aggregations per customer
# =========================
features = df.groupby("id").agg({
    "energie_kwh": ["mean", "std", "min", "max"]
})

features.columns = ["mean_energy", "std_energy", "min_energy", "max_energy"]


# =========================
# 4) Day vs Night consumption
# =========================
# Define night: 00:00 - 06:00
night_df = df[df["hour"] < 6]
day_df = df[df["hour"] >= 6]

night_consumption = night_df.groupby("id")["energie_kwh"].mean()
day_consumption = day_df.groupby("id")["energie_kwh"].mean()

features["night_mean"] = night_consumption
features["day_mean"] = day_consumption

features["night_day_ratio"] = features["night_mean"] / features["day_mean"]


# =========================
# 5) Week vs Weekend
# =========================
week_df = df[df["is_weekend"] == 0]
weekend_df = df[df["is_weekend"] == 1]

week_consumption = week_df.groupby("id")["energie_kwh"].mean()
weekend_consumption = weekend_df.groupby("id")["energie_kwh"].mean()

features["week_mean"] = week_consumption
features["weekend_mean"] = weekend_consumption

features["weekend_ratio"] = features["weekend_mean"] / features["week_mean"]


# =========================
# 6) Daily energy variability
# =========================
daily_energy = (
    df.groupby(["id", "date"])["energie_kwh"]
    .sum()
    .reset_index()
)

daily_stats = daily_energy.groupby("id")["energie_kwh"].agg(["mean", "std"])

features["daily_mean"] = daily_stats["mean"]
features["daily_std"] = daily_stats["std"]

features["daily_variability"] = features["daily_std"] / features["daily_mean"]


# =========================
# 7) Fill missing values
# =========================
features = features.fillna(0)


# =========================
# 8) Reset index
# =========================
features = features.reset_index()


# =========================
# 9) Save features
# =========================
features.to_csv(OUTPUT_PATH, index=False)

print("\nFeatures created:")
print(features.head())

print("\nSaved to:", OUTPUT_PATH)