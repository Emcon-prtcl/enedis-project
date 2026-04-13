from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Chemins vers les fichiers
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "outputs" / "sample_data.csv"
CLUSTER_PATH = BASE_DIR / "outputs" / "clustered_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


# Chargement des données
print("Loading data...")

df = pd.read_csv(DATA_PATH)
clusters = pd.read_csv(CLUSTER_PATH)


# On ajoute l'info de cluster à chaque ligne
df = df.merge(clusters[["id", "cluster"]], on="id")


# Création d'un index de temps (48 créneaux de 30 min sur une journée)
df["time_slot"] = df["hour"] * 2 + (df["minute"] // 30)


# Calcul de la courbe moyenne pour chaque cluster
avg_curve = df.groupby(["cluster", "time_slot"])["energie_kwh"].mean().reset_index()


# Génère une courbe "réaliste" en ajoutant un peu de bruit
def generate_curve(cluster_id):
    curve = avg_curve[avg_curve["cluster"] == cluster_id]["energie_kwh"].values
    
    # bruit proportionnel à la variabilité de la courbe
    noise = np.random.normal(0, curve.std() * 0.1, size=len(curve))
    synthetic = curve + noise
    
    return synthetic


# Génération de courbes pour les deux clusters
curve_0 = generate_curve(0)
curve_1 = generate_curve(1)


# Affichage des résultats
plt.figure(figsize=(10, 5))

plt.plot(curve_0, label="Cluster 0")
plt.plot(curve_1, label="Cluster 1")

plt.title("Generated consumption curves")
plt.xlabel("Time slot (30 min)")
plt.ylabel("Energy (kWh)")
plt.legend()

plot_path = OUTPUT_DIR / "generated_curves.png"
plt.savefig(plot_path)
plt.close()

print(f"\nSaved plot: {plot_path}")
