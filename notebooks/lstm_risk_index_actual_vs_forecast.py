"""Resolve the June/July-vs-NEMA-peak-months anomaly in the LSTM flood-risk index
(PROJECT_PLAN.md Phase 3, PAPER_DRAFT.md Section 5.3), without needing the NIHSA
river-discharge data that was requested but never received.

The percentile-rank flood-risk index (train_lstm.py) ranks each month's *forecast* VH
against the zone's training-period VH distribution. This script computes the same
percentile rank for the *actual observed* VH at the same test-period (2024-2025)
zone-months, against the same training-period reference distribution -- i.e. what the
risk index would have been with a perfect forecast. Comparing the two isolates whether
the anomaly is a real physical signal (would show up in both) or a forecasting
artifact (would show up only in the forecast)."""

import pandas as pd
from scipy.stats import percentileofscore

obs = pd.read_csv("data/lstm_zonal_timeseries.csv").dropna(subset=["VH"])
pred = pd.read_csv("outputs/tables/lstm_flood_risk_predictions.csv")

train = obs[obs["year"] <= 2023]

obs_test = obs.merge(pred[["zone", "year", "month"]], left_on=["ADM2_NAME", "year", "month"],
                      right_on=["zone", "year", "month"])


def actual_risk_index(row):
    zone_train = train[train["ADM2_NAME"] == row["ADM2_NAME"]]["VH"]
    return percentileofscore(zone_train, row["VH"], kind="mean") / 100


obs_test["actual_risk_index"] = obs_test.apply(actual_risk_index, axis=1)

merged = pred.merge(
    obs_test[["ADM2_NAME", "year", "month", "VH", "actual_risk_index"]],
    left_on=["zone", "year", "month"], right_on=["ADM2_NAME", "year", "month"]
)

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
by_month = merged.groupby("month")[["flood_risk_index", "actual_risk_index"]].mean().sort_index()
print("Month | LSTM-forecast risk index | Actual-observed risk index (same ref distribution)")
for m, row in by_month.iterrows():
    print(f"  {month_names[m - 1]}: forecast={row['flood_risk_index']:.3f}   actual={row['actual_risk_index']:.3f}")

corr = merged[["flood_risk_index", "actual_risk_index"]].corr().iloc[0, 1]
print(f"\nRow-level correlation (forecast vs actual, all zone-months): {corr:.3f}")

merged.to_csv("outputs/tables/lstm_forecast_vs_actual_risk_index.csv", index=False)
print("Saved outputs/tables/lstm_forecast_vs_actual_risk_index.csv")
