"""Export classified flood-extent GeoTIFFs (Objective 2 deliverable) for each documented
flood event (2019, 2020, 2022): retrain the RF classifier on pooled samples, classify
the full event-window SAR composite, and export the flood/no-flood classification band
to Drive as a GeoTIFF. This is the actual per-event flood map -- distinct from
build_frequency_map.py's 2015-2025 chronic-risk aggregate, and from
classify_2022_sanity_check.py, which computes area stats for the 2022 event but never
saves the classified image itself."""

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

EXPORT_SCALE = 60  # must match the scale training samples were extracted at
                    # (notebooks/collect_training_samples.py, SCALE=60) -- GLCM texture
                    # and the Refined Lee filter are fixed-pixel neighborhood windows,
                    # so classifying at a different analysis scale feeds the model a
                    # genuinely different feature distribution than it was trained on.
                    # (Confirmed: scale=20 gave ~74-79% "flooded" for events where the
                    # scale=60-consistent classify_2022_sanity_check.py gives 49.9%.)
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

tasks = []
for year in ["2019", "2020", "2022"]:
    e_start, e_end = FLOOD_EVENTS[year]
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
        description=f"bayelsa_{year}_flood_extent",
        folder=DRIVE_FOLDER,
        fileNamePrefix=f"bayelsa_{year}_flood_extent",
        region=aoi,
        scale=EXPORT_SCALE,
        crs="EPSG:4326",
        maxPixels=1e9,
    )
    task.start()
    tasks.append((year, task))
    print(f"Started export task: {year} flood extent (task id: {task.id})")

print("\nAll 3 export tasks started. Check status with earthengine task list, or:")
print("  https://code.earthengine.google.com/tasks")
print("Drive-exported files land in the GEE-linked Google account, not the claude.ai")
print("Drive connector's account -- download bayelsa_{2019,2020,2022}_flood_extent.tif")
print("from Drive manually once the tasks finish.")
