
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "energy_1hour_fe_ready.csv"
MODEL_PATH = BASE_DIR / "models" / "energy_pipeline.joblib"

# Original 24 PCA input features
FEATURES = (
    [f"T{i}" for i in range(1, 10)]
    + [f"RH_{i}" for i in range(1, 10)]
    + [
        "T_out",
        "RH_out",
        "Press_mm_hg",
        "Windspeed",
        "Visibility",
        "Tdewpoint"
    ]
)

TARGET = "Total_Energy"

# Load dataset
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Validate required columns
required = FEATURES + [TARGET]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

if df[required].isna().any().any():
    raise ValueError("Dataset contains missing values.")

# Chronological 80/20 split
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df[FEATURES]
X_test = test_df[FEATURES]

y_train = train_df[TARGET]
y_test = test_df[TARGET]

# Complete deployment pipeline
pipeline = Pipeline([
    ("input_scaler", StandardScaler()),
    ("pca", PCA(n_components=11)),
    ("pca_scaler", StandardScaler()),
    ("ridge", Ridge(alpha=100))
])

# Train model using training data only
pipeline.fit(X_train, y_train)

# Evaluation
train_pred = pipeline.predict(X_train)
test_pred = pipeline.predict(X_test)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\nDeployment Model Results")
print("Train RMSE:", round(
    np.sqrt(mean_squared_error(y_train, train_pred)), 3
))
print("Test MAE:", round(mean_absolute_error(y_test, test_pred), 3))
print("Test RMSE:", round(
    np.sqrt(mean_squared_error(y_test, test_pred)), 3
))
print("Test R2:", round(r2_score(y_test, test_pred), 3))

# Save pipeline and expected feature order
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "pipeline": pipeline,
        "features": FEATURES,
        "target": TARGET,
        "unit": "Wh"
    },
    MODEL_PATH
)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)
