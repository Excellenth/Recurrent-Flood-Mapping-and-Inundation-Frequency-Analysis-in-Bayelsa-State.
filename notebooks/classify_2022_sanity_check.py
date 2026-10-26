"""Retrain the RF (corrected feature set) and classify the full 2022 flood-event
composite, then check whether the estimated flooded area is physically plausible
against the known scale of the 2022 Bayelsa disaster (~1.3M displaced)."""

import sys
import json
import time

sys.path.insert(0, "src")

import ee
from gee_utils import init_ee, load_aoi_ee
from classification import FEATURE_BANDS, _despeckled_composite
from sar_processing import FLOOD_EVENTS
from features import build_feature_stack

init_ee()
aoi = load_aoi_ee()
dem = ee.Image("MERIT/DEM/v1_0_3")

records = []
for year in ["2019", "2020", "2022"]:
    with open(f"data/training_samples_{year}.json") as fh:
        records.extend(json.load(fh))

features = [ee.Feature(None, r) for r in records]
fc = ee.FeatureCollection(features)
classifier = ee.Classifier.smileRandomForest(numberOfTrees=200, seed=42).train(
    features=fc, classProperty="label", inputProperties=FEATURE_BANDS
)
print("Classifier retrained on", len(records), "samples with corrected features:", FEATURE_BANDS)

e_start, e_end = FLOOD_EVENTS["2022"]
event_img = _despeckled_composite(aoi, e_start, e_end)
full_stack = build_feature_stack(event_img, aoi, dem)

classified = full_stack.select(FEATURE_BANDS).classify(classifier)

t0 = time.time()
area_stats = classified.eq(1).multiply(ee.Image.pixelArea()).reduceRegion(
    reducer=ee.Reducer.sum(), geometry=aoi, scale=60, maxPixels=1e9, bestEffort=True
).getInfo()
total_area = aoi.area().getInfo()

flooded_km2 = area_stats["classification"] / 1e6
total_km2 = total_area / 1e6
pct_flooded = flooded_km2 / total_km2 * 100

print(f"Took {round(time.time()-t0,1)}s")
print(f"Total AOI area: {total_km2:.1f} km2")
print(f"Classified flooded area (2022 event composite): {flooded_km2:.1f} km2")
print(f"Percent of state classified as flooded: {pct_flooded:.1f}%")
