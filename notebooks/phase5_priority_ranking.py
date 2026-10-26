"""Objective 5: synthesize flood exposure (Phase 2), forward-looking risk trend
(Phase 3), and population exposure (WorldPop) into an LGA-level priority ranking for
flood-break embankment / intervention planning.

Scope note: this works at LGA granularity, not per-pixel. Pixel-resolution corridor
siting within the top LGAs is handled separately (phase5_embankment_siting.py), which
already uses the fully dual-masked frequency map.
"""

import pandas as pd

# 2022 event flood-classification % per LGA, JRC permanent-water AND SAR-derived
# persistent-wetland masked (from notebooks/lga_zonal_check_masked.py). This corrects
# an earlier JRC-only-masked version (Nembe 72.9, Brass 68.7, Southern Ijaw 55.9,
# Kolokuma/Opokuma 50.1, Yenegoa 43.3, Sagbama 40.3, Ekeremor 38.5, Ogbia 28.8) that
# left the swamp-forest double-bounce conflation uncorrected for the riverine/coastal
# LGAs in particular -- e.g. 54.3% of Nembe's and 50.9% of Brass's area was flagged as
# chronic wetland rather than 2022-specific flooding, versus only 16.7% of Yenegoa's.
flood_pct = {
    "Nembe": 38.5, "Kolokuma/Opokuma": 36.2, "Yenegoa": 33.7, "Brass": 32.1,
    "Sagbama": 27.7, "Southern Ijaw": 22.9, "Ekeremor": 20.8, "Ogbia": 19.4,
}

# WorldPop 2020 population sum per LGA
population = {
    "Yenegoa": 471073, "Southern Ijaw": 428499, "Ekeremor": 359065, "Sagbama": 248864,
    "Brass": 241072, "Ogbia": 240487, "Nembe": 174844, "Kolokuma/Opokuma": 106130,
}

# LSTM mean flood-risk-index, test period 2024-2025 (percentile-rank version)
lstm_risk = {
    "Kolokuma/Opokuma": 0.365, "Ogbia": 0.350, "Brass": 0.331, "Yenegoa": 0.328,
    "Sagbama": 0.320, "Southern Ijaw": 0.287, "Nembe": 0.255, "Ekeremor": 0.245,
}

df = pd.DataFrame({
    "LGA": list(flood_pct.keys()),
    "flood_pct_2022": list(flood_pct.values()),
}).set_index("LGA")
df["population"] = pd.Series(population)
df["lstm_risk_trend"] = pd.Series(lstm_risk)

# Est. population in flood-prone footprint -- the policy-relevant number for
# embankment prioritization (protecting people, not just land area).
df["est_population_exposed"] = (df["flood_pct_2022"] / 100 * df["population"]).round(0).astype(int)

# Composite priority score: normalize each component 0-1, weight population exposure
# highest (it's the direct human-impact metric), flood_pct and LSTM trend as
# supporting context (current severity + forward-looking trend).
def normalize(s):
    return (s - s.min()) / (s.max() - s.min())

df["priority_score"] = (
    0.5 * normalize(df["est_population_exposed"])
    + 0.3 * normalize(df["flood_pct_2022"])
    + 0.2 * normalize(df["lstm_risk_trend"])
).round(3)

df = df.sort_values("priority_score", ascending=False)
df.to_csv("outputs/tables/phase5_lga_priority_ranking.csv")

print(df.to_string())
print("\nSaved to outputs/tables/phase5_lga_priority_ranking.csv")
