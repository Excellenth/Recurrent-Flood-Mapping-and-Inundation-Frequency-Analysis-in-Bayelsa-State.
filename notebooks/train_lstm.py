"""Objective 3 prototype: LSTM forecasting next-month zonal VH backscatter (the
primary SAR flood-signature band, per the Phase 1 finding that both the open-water
decrease and flooded-vegetation increase show most clearly in VH) from a lookback
window of recent VV/VH/rainfall history, per LGA zone.

This predicts a continuous backscatter/flood-risk-trend signal, not a classified
flood extent map -- a deliberate scoping trade-off made to keep Phase 3 tractable
under the GEE compute quota restriction (see PROJECT_PLAN.md), not a hidden shortcut.

Train/test split is chronological (train through 2023, test 2024-2025), not random --
random splitting would leak future information into training for a time series.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras

LOOKBACK = 6
TEST_START_YEAR = 2024
FEATURES = ["VV", "VH", "rainfall_mm"]
TARGET = "VH"

df = pd.read_csv("data/lstm_zonal_timeseries.csv")
df = df.dropna(subset=["VV", "VH"]).sort_values(["ADM2_NAME", "year", "month"]).reset_index(drop=True)

# Build sliding-window sequences per zone.
X_list, y_list, meta_list = [], [], []
for zone, g in df.groupby("ADM2_NAME"):
    g = g.sort_values(["year", "month"]).reset_index(drop=True)
    values = g[FEATURES].values
    target_vals = g[TARGET].values
    dates = list(zip(g["year"], g["month"]))
    for i in range(LOOKBACK, len(g)):
        X_list.append(values[i - LOOKBACK:i])
        y_list.append(target_vals[i])
        meta_list.append((zone, dates[i][0], dates[i][1]))

X = np.array(X_list)
y = np.array(y_list)
meta = pd.DataFrame(meta_list, columns=["zone", "year", "month"])

print(f"Total sequences: {len(X)}  (shape {X.shape})")

train_mask = meta["year"] < TEST_START_YEAR
test_mask = ~train_mask
print(f"Train sequences: {train_mask.sum()}, Test sequences: {test_mask.sum()}")

X_train, X_test = X[train_mask.values], X[test_mask.values]
y_train, y_test = y[train_mask.values], y[test_mask.values]

# Fit scalers on TRAIN ONLY to avoid leaking test-period statistics.
n_features = X_train.shape[2]
scaler_X = StandardScaler().fit(X_train.reshape(-1, n_features))
X_train_s = scaler_X.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
X_test_s = scaler_X.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
y_train_s = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

model = keras.Sequential([
    keras.layers.Input(shape=(LOOKBACK, n_features)),
    keras.layers.LSTM(32, return_sequences=False),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

history = model.fit(
    X_train_s, y_train_s,
    validation_split=0.15,
    epochs=100,
    batch_size=16,
    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=0,
)
print(f"Training stopped after {len(history.history['loss'])} epochs")

y_pred_s = model.predict(X_test_s, verbose=0).ravel()
y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

# Naive baseline: predict "next month = same as last observed month" (persistence).
naive_pred = X_test[:, -1, FEATURES.index("VH")]
naive_mae = mean_absolute_error(y_test, naive_pred)
naive_rmse = mean_squared_error(y_test, naive_pred) ** 0.5

print(f"\n--- Test set (2024-2025), predicting next-month VH ---")
print(f"LSTM  MAE: {mae:.3f} dB, RMSE: {rmse:.3f} dB, R2: {r2:.3f}")
print(f"Naive persistence baseline MAE: {naive_mae:.3f} dB, RMSE: {naive_rmse:.3f} dB")
print(f"(LSTM should beat the naive baseline to be worth anything -- ", end="")
print("YES" if mae < naive_mae else "NO, LSTM did not beat the naive baseline", end=")\n")

model.save("outputs/models/lstm_zonal_vh_forecast.keras")
print("\nModel saved to outputs/models/lstm_zonal_vh_forecast.keras")
