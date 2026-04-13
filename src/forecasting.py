from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# Chemins vers les fichiers
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "outputs" / "daily_energy.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


# Chargement des données
print("Loading daily energy dataset...")
df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])


# On trie les données par client et par date
df = df.sort_values(by=["id", "date"])


# Création de variables basées sur les jours précédents
df["lag_1"] = df.groupby("id")["daily_energy_kwh"].shift(1)
df["lag_2"] = df.groupby("id")["daily_energy_kwh"].shift(2)

# Moyenne des 3 jours précédents (en décalant d'abord pour éviter la fuite de données)
df["rolling_mean_3"] = (
    df.groupby("id")["daily_energy_kwh"]
    .shift(1)
    .rolling(3)
    .mean()
    .reset_index(level=0, drop=True)
)


# On supprime les lignes incomplètes (liées aux lags)
df = df.dropna()

print("\nDataset after lag creation:")
print(df.head())


# Séparation des variables explicatives et de la cible
X = df[["lag_1", "lag_2", "rolling_mean_3"]]
y = df["daily_energy_kwh"]


# Split en train et test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Entraînement du modèle (régression linéaire simple)
model = LinearRegression()
model.fit(X_train, y_train)


# Prédictions sur le jeu de test
y_pred = model.predict(X_test)


# Évaluation avec l'erreur absolue moyenne
mae = mean_absolute_error(y_test, y_pred)

print("\nMean Absolute Error:", mae)


# Visualisation des valeurs réelles vs prédictions
plt.figure(figsize=(8, 4))
plt.scatter(y_test, y_pred)
plt.xlabel("Real values")
plt.ylabel("Predictions")
plt.title("Forecasting: Real vs Predicted")

plot_path = OUTPUT_DIR / "forecasting_scatter.png"
plt.savefig(plot_path)
plt.close()

print(f"\nSaved plot: {plot_path}")
