from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =========================
# 1) Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "outputs" / "clustered_data.csv"


# =========================
# 2) Load data
# =========================
print("Loading clustered data...")
df = pd.read_csv(DATA_PATH)

print("\nFirst rows:")
print(df.head())


# =========================
# 3) Prepare X and y
# =========================
X = df.drop(columns=["id", "cluster", "pca1", "pca2"])
y = df["cluster"]


# =========================
# 4) Train/Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)


# =========================
# 5) Logistic Regression
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)

print("\n" + "=" * 50)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 50)

print("\nAccuracy:", accuracy_score(y_test, y_pred_log))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))


# =========================
# 6) Random Forest
# =========================
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=5
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n" + "=" * 50)
print("RANDOM FOREST RESULTS")
print("=" * 50)

print("\nAccuracy:", accuracy_score(y_test, y_pred_rf))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))


# =========================
# 7) Feature importance
# =========================
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n" + "=" * 50)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 50)
print(importance_df)