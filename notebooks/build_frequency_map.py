"""Build the wet-season inundation frequency map (Objective 2 deliverable):
classify each wet-season (Aug-Nov) monthly composite from 2015-2025 with the trained
RF, then average across all 44 classified months to get, per pixel, the fraction of
wet-season months classified as flooded -- i.e. the chronic-flood-zone indicator.

Two layers get excluded from the final chronic-flood-RISK output, applied in sequence:

1. JRC permanent water (occurrence >= 50%, properly unmasked to 0 first -- JRC's
   occurrence band is nodata rather than 0 outside ever-observed-water pixels, which
   silently broke an earlier version of this mask). This catches permanent open
   water/rivers -- ~6% of the state.

2. SAR-derived "persistent wetland": pixels classified 'flood' in >=90% of the 44
   sampled wet-season months *after* the JRC mask is applied. Even with JRC water
   properly excluded, p95/p99 of the frequency distribution were still saturated at
   100% -- some pixels were classified 'flood' in essentially every one of 44
   independent wet-season observations across 11 years, which cannot be a genuine
   episodic flood event (those are tied to specific years, e.g. 2019/2020/2022, not
   every wet season for a decade). JRC's water detection is optical and can't see
   under forest canopy, so it misses flooded/swamp mangrove forest entirely -- exactly
   the double-bounce land cover identified in the Phase 1 validation as dominating
   this AOI. This second threshold is the SAR analog of JRC's own method (JRC defines
   "permanent" water the same way: consistency across many independent observations),
   just applied where optical detection fails. It is a post-classification temporal-
   consistency filter on the finished time series, not a retraining step, so it isn't
   circular in the way "filtering training data by model output" would be.

Both exclusion layers are kept as their own output bands for transparency about what
got removed, alongside the final flood_frequency band.

Submitted as a single async batch export since whole-image classification (full GLCM
feature stack) across 44 months is a much bigger computation than GEE's interactive
compute timeout allows for a synchronous call.
"""

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

EXPORT_SCALE = 60  # consistent with the scale used for training-sample extraction
DRIVE_FOLDER = "BayelsaFloodSAR"
WET_SEASON_MONTHS = [8, 9, 10, 11]
YEARS = range(2015, 2026)
PERMANENT_WATER_OCCURRENCE_THRESHOLD = 50
SAR_PERSISTENT_WETLAND_THRESHOLD = 0.90  # frequency, after JRC masking, at/above which
                                          # a pixel is treated as persistent wetland
                                          # rather than episodic flood risk

records = []
for year in ["2019", "2020", "2022"]:
    with open(f"data/training_samples_{year}.json") as fh:
        records.extend(json.load(fh))
features = [ee.Feature(None, r) for r in records]
fc = ee.FeatureCollection(features)
classifier = ee.Classifier.smileRandomForest(numberOfTrees=200, seed=42).train(
    features=fc, classProperty="label", inputProperties=FEATURE_BANDS
)
print("Classifier retrained on", len(records), "samples.")

classified_images = []
for year in YEARS:
    for month in WET_SEASON_MONTHS:
        start = f"{year}-{month:02d}-01"
        end = ee.Date(start).advance(1, "month")
        event_img = _despeckled_composite(aoi, start, end)
        full_stack = build_feature_stack(event_img, aoi, dem)
        classified = full_stack.select(FEATURE_BANDS).classify(classifier)
        classified_images.append(classified)

print("Built", len(classified_images), "monthly classified images (lazy).")

freq_collection = ee.ImageCollection.fromImages(classified_images)
frequency = freq_collection.mean().rename("flood_frequency")
n_valid = freq_collection.count().rename("n_valid_months")

permanent_water = (
    ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    .select("occurrence")
    .unmask(0)  # JRC occurrence is nodata (not 0) outside ever-observed-water pixels;
                # without this, .Not() below stays masked instead of becoming true for
                # dry land -- the same pitfall already caught once in features.py's
                # river_distance(), reintroduced here.
    .gte(PERMANENT_WATER_OCCURRENCE_THRESHOLD)
    .rename("permanent_water")
)
# Step 1: exclude JRC-visible permanent open water.
frequency_after_jrc = frequency.updateMask(permanent_water.Not())

# Step 2: on the remainder, flag pixels still persistently classified 'flood' across
# nearly all 44 months -- SAR-detected wetland that JRC's optical method missed under
# canopy (e.g. flooded mangrove/swamp forest).
sar_persistent_wetland = (
    frequency_after_jrc.gte(SAR_PERSISTENT_WETLAND_THRESHOLD).rename("sar_persistent_wetland")
)

frequency_final = frequency_after_jrc.updateMask(sar_persistent_wetland.Not()).rename("flood_frequency")
n_valid_final = n_valid.updateMask(permanent_water.Not()).updateMask(sar_persistent_wetland.Not())

output = (
    frequency_final
    .addBands(n_valid_final)
    .addBands(permanent_water)
    .addBands(sar_persistent_wetland)
    .clip(aoi)
)

task = ee.batch.Export.image.toDrive(
    image=output.toFloat(),
    description="bayelsa_wetseason_flood_frequency_2015_2025",
    folder=DRIVE_FOLDER,
    fileNamePrefix="bayelsa_wetseason_flood_frequency_2015_2025",
    region=aoi,
    scale=EXPORT_SCALE,
    crs="EPSG:4326",
    maxPixels=1e9,
)
task.start()
print("Started export task:", task.id)
