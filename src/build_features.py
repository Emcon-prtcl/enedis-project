from pathlib import Path
import pandas as pd
import numpy as np


# Chemins vers les fichiers
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "outputs" / "sample_data.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "features.csv"


# Chargement des données
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Reconvertit les colonnes de date après lecture du CSV
df["horodate"] = pd.to_datetime(df["horodate"], utc=True)
df["date"] = pd.to_datetime(df["date"])


# Statistiques de base par client
features = df.groupby("id").agg({
    "energie_kwh": ["mean", "std", "min", "max"]
})

features.columns = ["mean_energy", "std_energy", "min_energy", "max_energy"]


# Comparaison entre la consommation de nuit et de jour
night_df = df[df["hour"] < 6]
day_df = df[df["hour"] >= 6]

night_consumption = night_df.groupby("id")["energie_kwh"].mean()
day_consumption = day_df.groupby("id")["energie_kwh"].mean()

features["night_mean"] = night_consumption
features["day_mean"] = day_consumption
features["night_day_ratio"] = features["night_mean"] / features["day_mean"]


# Comparaison entre semaine et week-end
week_df = df[df["is_weekend"] == 0]
weekend_df = df[df["is_weekend"] == 1]

week_consumption = week_df.groupby("id")["energie_kwh"].mean()
weekend_consumption = weekend_df.groupby("id")["energie_kwh"].mean()

features["week_mean"] = week_consumption
features["weekend_mean"] = weekend_consumption
features["weekend_ratio"] = features["weekend_mean"] / features["week_mean"]


# Variabilité de la consommation journalière
daily_energy = (
    df.groupby(["id", "date"])["energie_kwh"]
    .sum()
    .reset_index()
)

daily_stats = daily_energy.groupby("id")["energie_kwh"].agg(["mean", "std"])

features["daily_mean"] = daily_stats["mean"]
features["daily_std"] = daily_stats["std"]
features["daily_variability"] = features["daily_std"] / features["daily_mean"]


# Remplacement des valeurs manquantes
features = features.fillna(0)


# Remet l'identifiant client comme colonne classique
features = features.reset_index()


# Sauvegarde des features
features.to_csv(OUTPUT_PATH, index=False)

print("\nFeatures created:")
print(features.head())

print("\nSaved to:", OUTPUT_PATH)
