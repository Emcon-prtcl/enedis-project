from pathlib import Path
import streamlit as st
import pandas as pd

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

FEATURES_PATH = OUTPUT_DIR / "features.csv"
CLUSTERED_PATH = OUTPUT_DIR / "clustered_data.csv"
DAILY_ENERGY_PATH = OUTPUT_DIR / "daily_energy.csv"

CLUSTERS_PCA_IMG = OUTPUT_DIR / "clusters_pca.png"
FORECASTING_IMG = OUTPUT_DIR / "forecasting_scatter.png"
GENERATED_IMG = OUTPUT_DIR / "generated_curves.png"
AVG_CURVE_IMG = OUTPUT_DIR / "average_daily_curve.png"
WEEKDAY_WEEKEND_IMG = OUTPUT_DIR / "weekday_vs_weekend_curve.png"

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Enedis Consumption Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Small custom CSS
# =========================
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
h1, h2, h3 {
    margin-top: 0.2rem;
}
.small-text {
    font-size: 0.95rem;
    color: #444;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Load data
# =========================
features_df = pd.read_csv(FEATURES_PATH)
clustered_df = pd.read_csv(CLUSTERED_PATH)
daily_energy_df = pd.read_csv(DAILY_ENERGY_PATH)
daily_energy_df["date"] = pd.to_datetime(daily_energy_df["date"])

# =========================
# Sidebar
# =========================
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Choisir une section",
    ["Introduction", "Clustering", "Classification", "Forecasting", "Generation"]
)

# =========================
# Header
# =========================
st.title("Enedis Consumption Analysis Dashboard")
st.markdown(
    '<div class="small-text">Projet de data science : clustering, classification, forecasting et génération de courbes.</div>',
    unsafe_allow_html=True
)
st.divider()

# =========================
# Section: Introduction
# =========================
if section == "Introduction":
    st.header("1. Introduction")

    st.write("""
    Ce dashboard présente une étude de profils de consommation électrique résidentielle.
    L'objectif est de :
    - regrouper les clients selon leur comportement,
    - prédire leur type à partir des features,
    - prévoir la consommation future,
    - générer des courbes synthétiques cohérentes.
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Nombre de clients", clustered_df["id"].nunique())
    c2.metric("Nombre de features", features_df.shape[1] - 1)
    c3.metric("Nombre de jours", daily_energy_df["date"].nunique())

    st.subheader("Aperçu des features")
    st.dataframe(features_df.head(10), use_container_width=True, height=280)

    st.subheader("Visualisations principales")
    col1, col2 = st.columns(2)

    with col1:
        st.image(str(AVG_CURVE_IMG), caption="Average daily curve", width=700)

    with col2:
        st.image(str(WEEKDAY_WEEKEND_IMG), caption="Weekday vs weekend curve", width=700)

# =========================
# Section: Clustering
# =========================
elif section == "Clustering":
    st.header("2. Clustering")

    st.write("""
    Nous avons utilisé K-Means avec k=2 pour identifier deux groupes de clients.
    Les données ont été normalisées puis projetées avec PCA pour visualiser les clusters.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Répartition")
        cluster_counts = clustered_df["cluster"].value_counts().sort_index()
        st.bar_chart(cluster_counts)

    with col2:
        st.subheader("Visualisation PCA")
        st.image(str(CLUSTERS_PCA_IMG), caption="K-Means clustering with PCA", width=700)

    st.subheader("Caractéristiques moyennes par cluster")
    cluster_means = clustered_df.groupby("cluster").mean(numeric_only=True)
    st.dataframe(cluster_means, use_container_width=True, height=260)

    st.info(
        "Interprétation proposée : un cluster correspond à des clients à plus forte consommation "
        "et plus grande variabilité (probablement RP), l’autre à des clients plus sobres et plus stables (probablement RS)."
    )

# =========================
# Section: Classification
# =========================
elif section == "Classification":
    st.header("3. Classification")

    st.write("""
    À partir des clusters obtenus, nous avons entraîné des modèles supervisés
    pour reproduire cette séparation.
    """)

    c1, c2 = st.columns(2)
    c1.metric("Logistic Regression", "0.95")
    c2.metric("Random Forest", "0.80")

    st.subheader("Interprétation")
    st.write("""
    La régression logistique obtient de meilleurs résultats que la Random Forest,
    ce qui suggère que les groupes sont relativement bien séparés par une frontière simple.
    """)

    st.subheader("Importance des variables (Random Forest)")
    importance_data = pd.DataFrame({
        "feature": [
            "week_mean", "mean_energy", "daily_mean", "weekend_mean",
            "night_mean", "day_mean", "daily_std", "std_energy",
            "weekend_ratio", "daily_variability", "night_day_ratio",
            "max_energy", "min_energy"
        ],
        "importance": [
            0.259730, 0.211583, 0.166377, 0.086048,
            0.066770, 0.059961, 0.059576, 0.029531,
            0.020963, 0.020789, 0.010517,
            0.008155, 0.000000
        ]
    })
    st.dataframe(importance_data, use_container_width=True, height=420)

# =========================
# Section: Forecasting
# =========================
elif section == "Forecasting":
    st.header("4. Forecasting")

    st.write("""
    Nous avons construit un modèle simple de prévision de la consommation journalière
    à partir de variables temporelles :
    - consommation de la veille,
    - consommation de l’avant-veille,
    - moyenne glissante.
    """)

    st.metric("Mean Absolute Error (MAE)", "3082 kWh")

    st.subheader("Real vs Predicted")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(str(FORECASTING_IMG), caption="Forecasting scatter plot", width=750)

    st.info(
        "Le modèle capture la tendance globale, mais reste limité car il ne prend pas en compte "
        "la météo, la saisonnalité détaillée ou d’autres variables externes."
    )

# =========================
# Section: Generation
# =========================
elif section == "Generation":
    st.header("5. Generation")

    st.write("""
    Nous avons généré des courbes synthétiques à partir des profils moyens des clusters,
    puis ajouté un bruit aléatoire pour reproduire une variabilité réaliste.
    """)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(str(GENERATED_IMG), caption="Generated consumption curves", width=750)

    st.info(
        "Les courbes générées reproduisent les tendances principales observées dans les données : "
        "un profil plus chargé pour les RP et plus modéré pour les RS."
    )