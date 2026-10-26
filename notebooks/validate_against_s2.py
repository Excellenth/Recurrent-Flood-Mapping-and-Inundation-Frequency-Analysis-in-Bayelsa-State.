"""Phase 4: independent validation of the SAR flood classification against Sentinel-2
NDWI for 2022-11-03 (within the 2022 flood event window, 14.5% cloud on the best tile).

Caveat kept in mind throughout: NDWI only detects open water (same canopy blind spot
as JRC), so this validates the open-water portion of the SAR classification, not the
flooded-vegetation double-bounce detection identified in Phase 1.
"""

import sys
import json

sys.path.insert(0, "src")

import ee
from gee_utils import init_ee, load_aoi_ee
from classification import FEATURE_BANDS, _despeckled_composite
from sar_processing import FLOOD_EVENTS
from features import build_feature_stack

init_ee()
aoi = load_aoi_ee()
dem = ee.Image("MERIT/DEM/v1_0_3")

S2_DATE = "2022-11-03"


def mask_s2_clouds(img):
    scl = img.select("SCL")
    # SCL classes: 3=cloud shadow, 8=cloud medium prob, 9=cloud high prob, 10=thin cirrus
    cloud_mask = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
    return img.updateMask(cloud_mask.Not())


s2_coll = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(aoi)
    .filterDate(S2_DATE, ee.Date(S2_DATE).advance(1, "day"))
    .map(mask_s2_clouds)
)
s2_mosaic = s2_coll.mosaic().clip(aoi)
ndwi = s2_mosaic.normalizedDifference(["B3", "B8"]).rename("NDWI")
s2_water = ndwi.gt(0.0).rename("s2_water")  # McFeeters threshold

# SAR classification for the matching event window
records = []
for year in ["2019", "2020", "2022"]:
    with open(f"data/training_samples_{year}.json") as fh:
        records.extend(json.load(fh))
features = [ee.Feature(None, r) for r in records]
fc = ee.FeatureCollection(features)
classifier = ee.Classifier.smileRandomForest(numberOfTrees=200, seed=42).train(
    features=fc, classProperty="label", inputProperties=FEATURE_BANDS
)

e_start, e_end = FLOOD_EVENTS["2022"]
event_img = _despeckled_composite(aoi, e_start, e_end)
full_stack = build_feature_stack(event_img, aoi, dem)
sar_classified = full_stack.select(FEATURE_BANDS).classify(classifier).rename("sar_flood")

# Compare only where both S2 (cloud-free) and SAR have valid data
comparison = sar_classified.addBands(s2_water)

# Sample points and build a confusion matrix client-side.
sample_pts = comparison.stratifiedSample(
    numPoints=1000, classBand="s2_water", region=aoi, scale=20, seed=1,
    geometries=False, tileScale=4, dropNulls=True,
)

info = sample_pts.getInfo()
print("Sample points retrieved:", len(info["features"]))

import collections
counts = collections.Counter()
for f in info["features"]:
    p = f["properties"]
    counts[(p["s2_water"], p["sar_flood"])] += 1

print("Confusion counts (s2_water, sar_flood): count")
for k, v in sorted(counts.items()):
    print(k, ":", v)

tp = counts[(1, 1)]
fp = counts[(0, 1)]
fn = counts[(1, 0)]
tn = counts[(0, 0)]
total = tp + fp + fn + tn
acc = (tp + tn) / total if total else float("nan")
print(f"\nAgreement (accuracy vs S2 open-water reference): {acc:.3f}  (n={total})")
print(f"SAR flood matches S2 water (TP): {tp}, SAR flood but S2 dry (FP): {fp}")
print(f"SAR non-flood but S2 water (FN, likely flooded veg S2 can't see or vice versa): {fn}, TN: {tn}")
