"""Export a per-pixel classified flood-extent GeoTIFF for 2024, real Sentinel-1 data
(not an LSTM forecast). 2024 falls inside the LSTM's 2024-2025 chronological test
window, which so far only has a zonal (LGA-level) risk index -- this fills in the
matching per-pixel map, same wet-season Sept 15-Nov 15 window and same pooled
classifier used for 2015/2016/2020/2022/2025, so 2024 is directly comparable to those.

2026-2030 is NOT treated this way: those years are genuinely in the future with no
real Sentinel-1 imagery to classify, so only the zonal LSTM baseline exists for them --
see PROJECT_PLAN.md Phase 3 and the risk_index_baseline_2026_2030 discussion."""

import sys
import json

sys.path.insert(0, "src")

import ee
from gee_utils import init_ee, load_aoi_ee
from classification import FEATURE_BANDS, _despeckled_composite
from features import build_feature_stack

init_ee()
aoi = load_aoi_ee()
dem = ee.Image("MERIT/DEM/v1_0_3")

EXPORT_SCALE = 60  # must match training scale -- see project memory / build_frequency_map.py
DRIVE_FOLDER = "BayelsaFloodSAR"
YEAR = "2024"
WINDOW = ("2024-09-15", "2024-11-15")

records = []
for year in ["2019", "2020", "2022"]:
    with open(f"data/training_samples_{year}.json") as fh:
        records.extend(json.load(fh))
features = [ee.Feature(None, r) for r in records]
fc = ee.FeatureCollection(features)
classifier = ee.Classifier.smileRandomForest(numberOfTrees=200, seed=42).train(
    features=fc, classProperty="label", inputProperties=FEATURE_BANDS
)
print("Classifier retrained on", len(records), "pooled samples.")

e_start, e_end = WINDOW
event_img = _despeckled_composite(aoi, e_start, e_end)
full_stack = build_feature_stack(event_img, aoi, dem)
classified = (
    full_stack.select(FEATURE_BANDS)
    .classify(classifier)
    .rename("flood_extent")
    .toByte()
    .clip(aoi)
)

task = ee.batch.Export.image.toDrive(
    image=classified,
    description=f"bayelsa_{YEAR}_flood_extent",
    folder=DRIVE_FOLDER,
    fileNamePrefix=f"bayelsa_{YEAR}_flood_extent",
    region=aoi,
    scale=EXPORT_SCALE,
    crs="EPSG:4326",
    maxPixels=1e9,
)
task.start()
print(f"Started export task: {YEAR} flood extent (task id: {task.id})")
print("\nDownload bayelsa_2024_flood_extent.tif from Drive once the task finishes")
print("(same account-mismatch limitation as always -- GEE-linked Drive account,")
print("not the claude.ai Drive connector's account).")
