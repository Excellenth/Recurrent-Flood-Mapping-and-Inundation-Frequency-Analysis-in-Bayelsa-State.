"""Export a classified flood-extent GeoTIFF for 2016, as a substitute for 2015 in the
5-year-interval series (2015/2020/2025 requested; 2015 excluded).

2014 has zero Sentinel-1 scenes over this AOI (mission had just launched, no coverage
yet). 2015 has data but its composite shows a systematic ~0.7-1.0 dB backscatter offset
from the 2019-2022 training-era baseline that persists regardless of window width
(tested Sept15-Nov15 and Aug1-Dec1, nearly identical offset both times, ruling out
sparse-scene noise as the cause) -- most likely an early-mission Sentinel-1 processing/
calibration difference the classifier was never trained to handle, not genuine ~80%
statewide flooding (implausible given even the worst documented event, 2022, was ~51%).
2016 (Sentinel-1B launched April 2016) is much closer to the training-era baseline
(VH/VV within ~0.2 dB of 2020's, vs. 2015's ~0.7-1.0 dB gap) and gives a plausible 57.1%
flooded estimate consistent with the other years' 50-55% range -- used here instead."""

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

EXPORT_SCALE = 60
DRIVE_FOLDER = "BayelsaFloodSAR"

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

event_img = _despeckled_composite(aoi, "2016-09-15", "2016-11-15")
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
    description="bayelsa_2016_flood_extent",
    folder=DRIVE_FOLDER,
    fileNamePrefix="bayelsa_2016_flood_extent",
    region=aoi,
    scale=EXPORT_SCALE,
    crs="EPSG:4326",
    maxPixels=1e9,
)
task.start()
print(f"Started export task: 2016 flood extent (task id: {task.id})")
print("Download bayelsa_2016_flood_extent.tif from Drive once the task finishes.")
