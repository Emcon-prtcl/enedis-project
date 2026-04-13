from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# Chemins vers les fichiers
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "outputs" / "sample_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# Chargement du dataset échantillon
print("Loading sample dataset...")
df = pd.read_csv(DATA_PATH)

print("\nFirst rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)


# Conversion des colonnes de date après rechargement du CSV
df["horodate"] = pd.to_datetime(df["horodate"], utc=True, errors="coerce")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("\nData types:")
print(df.dtypes)


# Vérification de la période couverte par les données
print("\nTime coverage:")
print("Start:", df["horodate"].min())
print("End  :", df["horodate"].max())

n_days = df["date"].nunique()
print("Number of distinct days:", n_days)


# Nombre de clients présents dans l'échantillon
n_customers = df["id"].nunique()
print("\nNumber of customers:", n_customers)


# Nombre total de mesures par client
counts_per_customer = df.groupby("id").size().sort_values()

print("\nMeasurements per customer:")
print(counts_per_customer.describe())

print("\nSmallest counts:")
print(counts_per_customer.head())

print("\nLargest counts:")
print(counts_per_customer.tail())


# Nombre de mesures par client et par jour
daily_counts = df.groupby(["id", "date"]).size()

print("\nMeasurements per customer per day:")
print(daily_counts.describe())

# Avec une mesure toutes les 30 minutes, on attend 48 points par jour
incomplete_days = daily_counts[daily_counts != 48]

print(f"\nNumber of incomplete customer-days (not equal to 48 points): {len(incomplete_days)}")


# Calcul de l'énergie journalière pour chaque client
daily_energy = (
    df.groupby(["id", "date"])["energie_kwh"]
    .sum()
    .reset_index()
    .rename(columns={"energie_kwh": "daily_energy_kwh"})
)

print("\nDaily energy dataset:")
print(daily_energy.head())

print("\nDaily energy stats:")
print(daily_energy["daily_energy_kwh"].describe())


# Visualisation de la consommation journalière pour un client
customer_id = df["id"].iloc[0]

customer_daily = daily_energy[daily_energy["id"] == customer_id].copy()
customer_daily = customer_daily.sort_values("date")

plt.figure(figsize=(10, 4))
plt.plot(customer_daily["date"], customer_daily["daily_energy_kwh"])
plt.title(f"Daily energy consumption - customer {customer_id}")
plt.xlabel("Date")
plt.ylabel("Daily energy (kWh)")
plt.xticks(rotation=45)
plt.tight_layout()
plot1_path = OUTPUT_DIR / "daily_energy_one_customer.png"
plt.savefig(plot1_path)
plt.close()

print(f"\nSaved plot: {plot1_path}")


# Courbe moyenne sur une journée
df["time_slot"] = df["hour"].astype(str).str.zfill(2) + ":" + df["minute"].astype(str).str.zfill(2)

mean_curve = (
    df.groupby("time_slot")["energie_kwh"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(10, 4))
plt.plot(mean_curve["time_slot"], mean_curve["energie_kwh"])
plt.title("Average energy by 30-minute time slot")
plt.xlabel("Time slot")
plt.ylabel("Average energy (kWh)")
plt.xticks(rotation=90)
plt.tight_layout()
plot2_path = OUTPUT_DIR / "average_daily_curve.png"
plt.savefig(plot2_path)
plt.close()

print(f"Saved plot: {plot2_path}")


# Comparaison entre semaine et week-end
weekday_weekend = (
    df.groupby(["is_weekend", "time_slot"])["energie_kwh"]
    .mean()
    .reset_index()
)

weekday_curve = weekday_weekend[weekday_weekend["is_weekend"] == 0]
weekend_curve = weekday_weekend[weekday_weekend["is_weekend"] == 1]

plt.figure(figsize=(10, 4))
plt.plot(weekday_curve["time_slot"], weekday_curve["energie_kwh"], label="Weekday")
plt.plot(weekend_curve["time_slot"], weekend_curve["energie_kwh"], label="Weekend")
plt.title("Weekday vs weekend average curve")
plt.xlabel("Time slot")
plt.ylabel("Average energy (kWh)")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plot3_path = OUTPUT_DIR / "weekday_vs_weekend_curve.png"
plt.savefig(plot3_path)
plt.close()

print(f"Saved plot: {plot3_path}")


# Sauvegarde du dataset d'énergie journalière
daily_energy_path = OUTPUT_DIR / "daily_energy.csv"
daily_energy.to_csv(daily_energy_path, index=False)
print(f"\nSaved daily energy dataset: {daily_energy_path}")
