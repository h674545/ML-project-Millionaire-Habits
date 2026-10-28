import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import joblib

# --- 1. Last inn data ---
print("Laster inn data...")
df = pd.read_csv("archive/Millionaire_Habits_Dataset.csv")
print(f"  Totalt antall rader: {len(df):,}")

# --- 2. Velg features og label ---
FEATURES = ["age", "total_annual_income", "savings_rate"]
TARGET = "millionaire_status"

df_model = df[FEATURES + [TARGET]].dropna()
print(f"  Rader etter dropna: {len(df_model):,}")
print(f"  Andel millionærer:  {df_model[TARGET].mean():.1%}")

# --- 3. Del opp i trening og test ---
X = df_model[FEATURES]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)
print(f"\nTreningssett: {len(X_train):,} rader")
print(f"Testsett:     {len(X_test):,} rader")

# --- 4. Bygg Pipeline ---
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )),
])

# --- 5. Tren ---
print("\nTrener modellen...")
pipeline.fit(X_train, y_train)

# --- 6. Evaluer ---
y_prob = pipeline.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(f"\nROC-AUC på testsett: {auc:.4f}")

if auc > 0.97:
    print("⚠️  AUC er veldig høy – sjekk for data leakage!")
elif auc >= 0.80:
    print("✓  AUC ser fornuftig ut")

# --- 7. Lagre ---
joblib.dump(pipeline, "model.joblib")
print("Modell lagret til model.joblib")
