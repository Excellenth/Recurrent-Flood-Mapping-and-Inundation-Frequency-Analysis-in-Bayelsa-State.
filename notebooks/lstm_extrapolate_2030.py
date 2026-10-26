"""Extrapolate the LSTM flood-risk index forward to 2030 as a stable, no-trend
"business-as-usual" baseline (requested to connect with SDG 2030 target-year framing --
the framing itself is a separate, later step; this is just the number-generating
extrapolation).

First version of this recursively fed the model's own prior VH forecast back in as the
VH lookback input for the next month. That compounded: for the identical July
climatology every year, predicted VH rose monotonically 2026->2030 (-14.20 -> -13.93
dB) with no external driver to explain it -- a classic recursive-forecast drift
artifact, not a real trend, and was discarded rather than reported.

This version removes the recursion entirely: each future month's 6-month lookback
window is built ONLY from each zone's own historical (2015-2025) per-calendar-month
climatology for ALL THREE features (VV, VH, rainfall) -- never from a prior model
prediction. Since the same lookback recurs identically every year for a given calendar
month, the model's output for that month is identical every year by construction: a
flat, non-drifting repeating annual cycle, representing "if the 2015-2025 seasonal
pattern simply repeats with no trend." This is deliberately NOT a genuine forecast of
what 2030 will actually look like -- it does not incorporate climate-change-driven
rainfall intensification, sea-level rise, or land-use change, all material factors for
the Niger Delta specifically. It is a baseline: any real future change, or the effect of
the interventions recommended in Objective 5, would show up as a departure from this
flat line, not as a change to it."""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

LOOKBACK = 6
FEATURES = ["VV", "VH", "rainfall_mm"]
TARGET = "VH"
TEST_START_YEAR = 2024
YEARS = range(2026, 2031)

df = pd.read_csv("data/lstm_zonal_timeseries.csv").dropna(subset=["VV", "VH"])
df = df.sort_values(["ADM2_NAME", "year", "month"]).reset_index(drop=True)

train_df = df[df["year"] < TEST_START_YEAR]
scaler_X = StandardScaler().fit(train_df[FEATURES].values)
scaler_y = StandardScaler().fit(train_df[[TARGET]].values)
historical_vh = {zone: g["VH"].values for zone, g in train_df.groupby("ADM2_NAME")}

# Per-zone, per-calendar-month climatology for ALL THREE features (full 2015-2025
# record) -- this is what stands in for "typical" future conditions.
clim = df.groupby(["ADM2_NAME", "month"])[FEATURES].mean()

model = keras.models.load_model("outputs/models/lstm_zonal_vh_forecast.keras")

results = []
for zone in df["ADM2_NAME"].unique():
    for target_month in range(1, 13):
        # 6-month lookback of pure climatology ending the month before target_month.
        lookback_months = [(target_month - 1 - k - 1) % 12 + 1 for k in range(LOOKBACK)][::-1]
        window = np.array([clim.loc[(zone, mm), FEATURES].values for mm in lookback_months])

        X_in = scaler_X.transform(window).reshape(1, LOOKBACK, len(FEATURES))
        pred_s = model.predict(X_in, verbose=0).ravel()[0]
        pred_vh = scaler_y.inverse_transform([[pred_s]])[0, 0]

        hist = historical_vh[zone]
        risk = float((hist < pred_vh).mean())

        for year in YEARS:
            results.append({"zone": zone, "year": year, "month": target_month,
                             "predicted_VH": pred_vh, "flood_risk_index": risk})

out = pd.DataFrame(results)
out.to_csv("outputs/tables/lstm_extrapolation_2026_2030.csv", index=False)

print("Stability check -- July predicted VH by year (should be identical, no drift):")
print(out[(out["zone"] == "Yenegoa") & (out["month"] == 7)][["year", "predicted_VH"]].to_string(index=False))

print("\nMean flood-risk index by zone, baseline (identical every year 2026-2030):")
print(out.groupby("zone")["flood_risk_index"].mean().sort_values(ascending=False).to_string())

print("\nMean flood-risk index by month, baseline:")
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
by_month = out.groupby("month")["flood_risk_index"].mean()
for m, v in by_month.items():
    print(f"  {month_names[m-1]}: {v:.3f}")

print("\nSaved outputs/tables/lstm_extrapolation_2026_2030.csv")
