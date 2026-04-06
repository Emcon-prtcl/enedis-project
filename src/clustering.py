from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# =========================
# 1) Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "outputs" / "features.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


# =========================
# 2) Load features
# =========================
print("Loading features...")
df = pd.read_csv(DATA_PATH)

print("\nFirst rows:")
print(df.head())


# =========================
# 3) Prepare data
# =========================
X = df.drop(columns=["id"])

# Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nData normalized.")


# =========================
# 4) K-Means clustering
# =========================
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["cluster"] = clusters

print("\nCluster distribution:")
print(df["cluster"].value_counts())


# =========================
# 5) PCA for visualization
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]


# =========================
# 6) Plot clusters
# =========================
plt.figure(figsize=(6, 5))
plt.scatter(df["pca1"], df["pca2"], c=df["cluster"])
plt.title("K-Means clustering (k=2)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plot_path = OUTPUT_DIR / "clusters_pca.png"
plt.savefig(plot_path)
plt.close()

print(f"\nSaved plot: {plot_path}")


# =========================
# 7) Cluster analysis
# =========================
cluster_means = df.groupby("cluster").mean()

print("\nCluster characteristics:")
print(cluster_means)


# =========================
# 8) Save results
# =========================
output_path = OUTPUT_DIR / "clustered_data.csv"
df.to_csv(output_path, index=False)

print(f"\nSaved clustered data: {output_path}")