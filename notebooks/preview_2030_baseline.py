"""Figure for the 2026-2030 baseline projection (notebooks/lstm_extrapolate_2030.py):
the stable, repeating seasonal flood-risk-index cycle used as the no-trend baseline."""

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("outputs/tables/lstm_extrapolation_2026_2030.csv")
by_month = df.groupby("month")["flood_risk_index"].mean().sort_index()
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(range(1, 13), by_month.values, marker="o", color="#2b6cb0", linewidth=2)
ax.axvspan(6, 11, alpha=0.08, color="#c53030")
ax.text(8.5, 0.05, "NEMA-reported\nwet season", ha="center", fontsize=9, color="#c53030")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)
ax.set_ylabel("Flood-risk index (baseline)")
ax.set_ylim(0, 1)
ax.set_title("2026-2030 no-trend baseline: mean statewide flood-risk index by month")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("outputs/figures/baseline_2030_seasonal_cycle.png", dpi=200)
print("Saved outputs/figures/baseline_2030_seasonal_cycle.png")
