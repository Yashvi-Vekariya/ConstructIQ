# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import os

# === Paths ===
DATA_DIR = "data/cleaned"
TRAIN_FILE = os.path.join(DATA_DIR, "train_dataset.csv")
VALID_FILE = os.path.join(DATA_DIR, "validation_dataset.csv")
MODEL_FILE = "construction_cost_model.pkl"

# === Load Data ===
print("📂 Loading datasets...")
train_df = pd.read_csv(TRAIN_FILE)
valid_df = pd.read_csv(VALID_FILE)

# === Combine Train + Validation ===
df = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)

# === Target column (UPDATE this if needed) ===
TARGET_COL = "total_cost"  # <-- change to actual target column name
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

# === Features / Target split ===
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# === One-hot encode categorical features ===
X_encoded = pd.get_dummies(X, drop_first=True)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# === Scale numeric features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Train Model ===
print("🚀 Training model...")
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# === Predictions & Evaluation ===
y_pred = model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained successfully!")
print(f"📊 RMSE: {rmse:.2f}")
print(f"📈 R² Score: {r2:.4f}")

# === Save Model & Scaler ===
joblib.dump({"model": model, "scaler": scaler, "columns": X_encoded.columns.tolist()}, MODEL_FILE)
print(f"💾 Model saved to {MODEL_FILE}")
