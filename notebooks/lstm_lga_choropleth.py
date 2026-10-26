"""Build the LSTM flood-risk index as an actual spatial map, not just a table.

The LSTM (train_lstm.py) forecasts a zonal (LGA-level) time series, not a per-pixel
classification -- there is no LSTM raster by design (Section 3.6). But the zonal risk
index can still be joined to the LGA polygons already in hand
(data/aoi/bayelsa_lgas_all.geojson) to produce a proper choropleth map, which did not
exist anywhere in the project until now. Two panels: the real 2024-2025 test-period mean
risk index (Table 3a), and the 2026-2030 no-trend baseline mean (Section 4.4)."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import geopandas as gpd

lgas = gpd.read_file("data/aoi/bayelsa_lgas_all.geojson")

test_period = pd.read_csv("outputs/tables/lstm_flood_risk_predictions.csv")
test_mean = test_period.groupby("zone")["flood_risk_index"].mean().rename("risk_index_test_2024_2025")

baseline = pd.read_csv("outputs/tables/lstm_extrapolation_2026_2030.csv")
baseline_mean = baseline.groupby("zone")["flood_risk_index"].mean().rename("risk_index_baseline_2026_2030")

lgas = lgas.merge(test_mean, left_on="ADM2_NAME", right_index=True, how="left")
lgas = lgas.merge(baseline_mean, left_on="ADM2_NAME", right_index=True, how="left")

missing = lgas[lgas["risk_index_test_2024_2025"].isna()]
if len(missing):
    raise ValueError(f"Unmatched LGA names: {missing['ADM2_NAME'].tolist()}")

lgas.to_file("data/aoi/bayelsa_lgas_lstm_risk.geojson", driver="GeoJSON")
print("Saved data/aoi/bayelsa_lgas_lstm_risk.geojson")

cmap = mcolors.LinearSegmentedColormap.from_list("risk", ["#f7fafc", "#c53030"])
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, col, title in zip(
    axes,
    ["risk_index_test_2024_2025", "risk_index_baseline_2026_2030"],
    ["LSTM flood-risk index\n(test period, 2024-2025 mean)", "2026-2030 no-trend baseline\n(mean)"],
):
    lgas.plot(column=col, ax=ax, cmap=cmap, vmin=0.2, vmax=0.55, edgecolor="black", linewidth=0.5, legend=True)
    for _, row in lgas.iterrows():
        c = row["geometry"].representative_point()
        ax.annotate(f"{row['ADM2_NAME']}\n{row[col]:.3f}", (c.x, c.y), ha="center", fontsize=7)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("LSTM zonal flood-risk index by LGA", fontsize=13)
fig.tight_layout()
fig.savefig("outputs/figures/lstm_lga_risk_choropleth.png", dpi=200)
print("Saved outputs/figures/lstm_lga_risk_choropleth.png")
