from pathlib import Path
import pandas as pd


# =========================
# 1) Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "export.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "sample_data.csv"


# =========================
# 2) Load only useful columns
# =========================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, usecols=["id", "horodate", "valeur"])

print("\nFirst rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# =========================
# 3) Convert timestamp
# =========================
df["horodate"] = pd.to_datetime(df["horodate"], errors="coerce", utc=True)

print("\nData types after datetime conversion:")
print(df.dtypes)


# =========================
# 4) Remove invalid rows
# =========================
df = df.dropna(subset=["id", "horodate", "valeur"]).copy()

print("\nShape after dropping missing values:")
print(df.shape)


# =========================
# 5) Sort values
# =========================
df = df.sort_values(by=["id", "horodate"]).reset_index(drop=True)


# =========================
# 6) Convert power to energy
# =========================
# The dataset gives power values every 30 minutes.
# Energy over 30 min = power * 0.5 hour
df["energie_kwh"] = df["valeur"] * 0.5


# =========================
# 7) Create useful temporal columns
# =========================
df["date"] = df["horodate"].dt.date
df["hour"] = df["horodate"].dt.hour
df["minute"] = df["horodate"].dt.minute
df["day_of_week"] = df["horodate"].dt.dayofweek  # Monday=0, Sunday=6
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)


# =========================
# 8) Check how many customers we have
# =========================
n_customers = df["id"].nunique()
print(f"\nNumber of unique customers: {n_customers}")


# =========================
# 9) Create a smaller sample for development
# =========================
# We keep only the first 100 customers to make development faster.
sample_ids = df["id"].drop_duplicates().head(100)
sample_df = df[df["id"].isin(sample_ids)].copy()

print("\nSample dataset shape:")
print(sample_df.shape)

print("\nSample customers:")
print(sample_df["id"].nunique())


# =========================
# 10) Save sample
# =========================
sample_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSample saved to: {OUTPUT_PATH}")