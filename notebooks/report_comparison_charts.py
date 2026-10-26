"""Two summary charts for FORECAST_COMPARISON_REPORT.md: flooded vs. non-flooded %
by classified year, and observed vs. projected LSTM risk index by LGA. Numbers are
copied from already-published sources (flood_extent_comparison_2019_2020_2022.png,
flood_extent_5yr_interval_2016_2020_2025.png, data/aoi/bayelsa_lgas_lstm_risk.geojson)
rather than recomputed, so they match the paper and the earlier Claude Docs report
exactly."""

import matplotlib.pyplot as plt
import numpy as np

# --- Chart 1: flooded vs. non-flooded % by year (stacked, sums to 100%) ---
years = ["2016", "2019", "2020", "2022", "2025"]
flooded = [58.6, 50.5, 54.5, 51.3, 51.2]
non_flooded = [100 - f for f in flooded]
documented = [False, True, True, True, False]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(years))
ax.bar(x, flooded, color="#c53030", label="Flooded")
ax.bar(x, non_flooded, bottom=flooded, color="#2b6cb0", label="Non-flooded")
for i, f in enumerate(flooded):
    ax.text(i, f - 4, f"{f:.1f}%", ha="center", va="top", color="white", fontsize=10, fontweight="bold")
labels = [f"{y}\n(event)" if d else f"{y}\n(snapshot)" for y, d in zip(years, documented)]
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("% of Bayelsa State")
ax.set_ylim(0, 100)
ax.set_title("Classified flood extent: a narrow 50.5–58.6% band across all five mapped years")
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig("outputs/figures/report_year_flood_share.png", dpi=200)
print("Saved outputs/figures/report_year_flood_share.png")

# --- Chart 2: observed vs. projected LSTM risk index by LGA ---
lgas = ["Ekeremor", "Nembe", "Southern Ijaw", "Brass", "Sagbama", "Ogbia", "Yenegoa", "Kolokuma/Opokuma"]
observed = [0.245, 0.255, 0.287, 0.331, 0.320, 0.350, 0.328, 0.365]
projected = [0.489, 0.481, 0.511, 0.510, 0.499, 0.502, 0.479, 0.485]

fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(lgas))
w = 0.35
ax.bar(x - w / 2, observed, w, color="#a0aec0", label="Observed 2024–2025")
ax.bar(x + w / 2, projected, w, color="#2b6cb0", label="Projected 2026–2030")
ax.set_xticks(x)
ax.set_xticklabels(lgas, rotation=30, ha="right")
ax.set_ylabel("Flood-risk index (0–1)")
ax.set_title("Projected flood-risk index rises in every LGA, +0.12 to +0.24")
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig("outputs/figures/report_lga_risk_forecast.png", dpi=200)
print("Saved outputs/figures/report_lga_risk_forecast.png")
