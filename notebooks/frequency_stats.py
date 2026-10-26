"""Recompute the wet-season flood frequency image (same graph as build_frequency_map.py)
and export summary statistics to a GEE table ASSET (not Drive) so results can be read
back directly via the EE API without needing Drive access."""

import sys
import json

sys.path.insert(0, "src")

import ee
from gee_utils import init_ee, load_aoi_ee
from classification import FEATURE_BANDS, _despeckled_composite
from features import build_feature_stack

PROJECT = "ee-seunlekan"

init_ee()
aoi = load_aoi_ee()
dem = ee.Image("MERIT/DEM/v1_0_3")

WET_SEASON_MONTHS = [8, 9, 10, 11]
YEARS = range(2015, 2026)

records = []
for year in ["2019", "2020", "2022"]:
    with open(f"data/training_samples_{year}.json") as fh:
        records.extend(json.load(fh))
features = [ee.Feature(None, r) for r in records]
fc = ee.FeatureCollection(features)
classifier = ee.Classifier.smileRandomForest(numberOfTrees=200, seed=42).train(
    features=fc, classProperty="label", inputProperties=FEATURE_BANDS
)

classified_images = []
for year in YEARS:
    for month in WET_SEASON_MONTHS:
        start = f"{year}-{month:02d}-01"
        end = ee.Date(start).advance(1, "month")
        event_img = _despeckled_composite(aoi, start, end)
        full_stack = build_feature_stack(event_img, aoi, dem)
        classified = full_stack.select(FEATURE_BANDS).classify(classifier)
        classified_images.append(classified)

freq_collection = ee.ImageCollection.fromImages(classified_images)
frequency = freq_collection.mean().rename("flood_frequency")

permanent_water = (
    ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).gte(50)
).rename("permanent_water")
frequency_after_jrc = frequency.updateMask(permanent_water.Not())

# SAR-derived persistent wetland: pixels still classified 'flood' in >=90% of the 44
# months even after excluding JRC's (optical, canopy-blind) permanent water. See
# build_frequency_map.py for the full rationale.
sar_persistent_wetland = (
    frequency_after_jrc.gte(0.90).rename("sar_persistent_wetland")
)
frequency_final = frequency_after_jrc.updateMask(sar_persistent_wetland.Not()).rename("flood_frequency")

# Include the exclusion layers unmasked in the same multi-band image/reduceRegion call
# so we get their area fractions too, without extra reduceRegion passes over the
# expensive 44-month graph (each pass seems to force independent re-evaluation, which
# caused an earlier out-of-memory failure).
stats_image = frequency_final.addBands(permanent_water).addBands(sar_persistent_wetland).clip(aoi)

# Single combined reducer, single reduceRegion call -- the earlier version ran 4
# separate reduceRegion calls on the same frequency image, and each one apparently
# re-evaluated the entire 44-month classification graph from scratch rather than
# reusing a shared result, which is what caused the out-of-memory failure.
# (fixedHistogram was tried here too, but its 2D array output can't be encoded as a
# table/feature property for export -- scalars only.)
combined_reducer = (
    ee.Reducer.percentile([10, 25, 50, 75, 90, 95, 99])
    .combine(ee.Reducer.mean(), sharedInputs=True)
    .combine(ee.Reducer.minMax(), sharedInputs=True)
)

stats = stats_image.reduceRegion(
    reducer=combined_reducer,
    geometry=aoi, scale=500, maxPixels=1e9, bestEffort=True,
)

summary = ee.Dictionary(stats).combine(ee.Dictionary({"total_area_m2": aoi.area()}))
# Export.table requires non-null geometry (see the same failure mode hit earlier with
# training-sample exports) -- the geometry's value doesn't matter here, only the
# properties, so a throwaway centroid point satisfies the requirement.
stats_fc = ee.FeatureCollection([ee.Feature(aoi.centroid(), summary)])

asset_id = f"projects/{PROJECT}/assets/bayelsa_frequency_stats"
task = ee.batch.Export.table.toAsset(
    collection=stats_fc,
    description="bayelsa_frequency_stats",
    assetId=asset_id,
)
task.start()
print("Started export task:", task.id, "-> asset:", asset_id)
