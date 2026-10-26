"""Turn the LSTM's raw next-month VH forecast into an interpretable per-zone flood-risk
index, closing the loop on Objective 3 ("predicting future flood extents").

First version of this used a two-point benchmark (mean VH during known flood months
vs. mean VH during dry-season baseline months, ~6-9 data points each per zone) and
scaled forecasts linearly between them. That gap turned out to be only 0.19-0.35 dB
per zone -- smaller than the LSTM's own prediction error (MAE 0.116 dB) -- so the
index was mostly amplifying noise: June (dry/transitional season) scored higher than
September-November (actual flood season), which isn't physically sensible.

This version instead uses each zone's percentile rank within its own FULL historical
VH distribution (training period only, 2015-2023, ~120 months/zone) as the flood-risk
index. Using the whole distribution rather than ~6-9 flood-only points is far less
noise-sensitive, and "how unusual is this forecast relative to this zone's own normal
range" is arguably a more defensible definition of risk anyway. All 8 zones show a
flood = VH increase signature (consistent with Phase 1's double-bounce finding), so a
higher percentile rank consistently means higher flood risk across zones.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

LOOKBACK = 6
TEST_START_YEAR = 2024
FEATURES = ["VV", "VH", "rainfall_mm"]
TARGET = "VH"

df = pd.read_csv("data/lstm_zonal_timeseries.csv")
df = df.dropna(subset=["VV", "VH"]).sort_values(["ADM2_NAME", "year", "month"]).reset_index(drop=True)

# Per-zone historical VH distribution, TRAINING PERIOD ONLY (2015-2023) -- using only
# the past as the reference distribution for interpreting a forecast is both more
# defensible (no peeking at the test period) and more meaningful ("how unusual is
# this forecast relative to what this zone has seen before").
historical_vh = {
    zone: g[g["year"] < TEST_START_YEAR]["VH"].values
    for zone, g in df.groupby("ADM2_NAME")
}
print("Per-zone historical VH distribution (training period, n months):")
for zone, vals in historical_vh.items():
    print(f"  {zone}: n={len(vals)}, range=[{vals.min():.2f}, {vals.max():.2f}] dB")

# Rebuild the same sequences/split as train_lstm.py so predictions line up with metadata.
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

train_mask = meta["year"] < TEST_START_YEAR
test_mask = ~train_mask
X_train, X_test = X[train_mask.values], X[test_mask.values]
y_train = y[train_mask.values]

n_features = X_train.shape[2]
scaler_X = StandardScaler().fit(X_train.reshape(-1, n_features))
X_test_s = scaler_X.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

model = keras.models.load_model("outputs/models/lstm_zonal_vh_forecast.keras")
y_pred_s = model.predict(X_test_s, verbose=0).ravel()
y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

meta_test = meta[test_mask.values].reset_index(drop=True)
meta_test["predicted_VH"] = y_pred


def risk_index(row):
    """Percentile rank (0-1) of the predicted VH within this zone's historical
    training-period distribution -- higher = more unusual/elevated relative to that
    zone's own normal range, consistent with the flood=VH-increase signature found
    for every zone."""
    hist = historical_vh[row["zone"]]
    return float((hist < row["predicted_VH"]).mean())


meta_test["flood_risk_index"] = meta_test.apply(risk_index, axis=1)
meta_test.to_csv("outputs/tables/lstm_flood_risk_predictions.csv", index=False)

print(f"\nSaved {len(meta_test)} forecast + flood-risk-index rows to outputs/tables/lstm_flood_risk_predictions.csv")
print("\nHighest flood-risk-index forecasts (test period, 2024-2025):")
print(meta_test.sort_values("flood_risk_index", ascending=False).head(10).to_string(index=False))

print("\nMean flood-risk-index by zone (test period):")
print(meta_test.groupby("zone")["flood_risk_index"].mean().sort_values(ascending=False).to_string())

print("\nMean flood-risk-index by month (test period, seasonal pattern check):")
print(meta_test.groupby("month")["flood_risk_index"].mean().sort_values(ascending=False).to_string())
