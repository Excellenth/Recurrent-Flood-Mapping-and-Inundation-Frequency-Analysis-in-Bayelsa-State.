"""Extract the zonal monthly SAR + rainfall time series (2015-2025) for LSTM training.
Pulled one year at a time (12 months x 8 LGAs per call) rather than the whole 2015-2025
range in one request -- smaller batches proved more reliable against GEE's interactive
compute timeout during Phase 2, and this keeps each call small even under the current
quota-restricted throughput."""

import sys
import csv

sys.path.insert(0, "src")

import ee
from gee_utils import init_ee, load_aoi_ee
from lstm_data import zonal_month_features, bayelsa_lgas

init_ee()
aoi = load_aoi_ee()
lgas = bayelsa_lgas()

YEARS = range(2015, 2026)
OUT_PATH = "data/lstm_zonal_timeseries.csv"

all_rows = []
for year in YEARS:
    year_fc = None
    for month in range(1, 13):
        monthly = zonal_month_features(aoi, year, month, lgas=lgas)
        year_fc = monthly if year_fc is None else year_fc.merge(monthly)

    info = year_fc.getInfo()
    rows = [f["properties"] for f in info["features"]]
    all_rows.extend(rows)
    print(f"{year}: {len(rows)} zone-months retrieved")

fieldnames = ["year", "month", "ADM2_NAME", "VV", "VH", "rainfall_mm"]
with open(OUT_PATH, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for r in all_rows:
        writer.writerow(r)

print(f"\nSaved {len(all_rows)} rows to {OUT_PATH}")
