"""Export classified flood-extent GeoTIFFs for 2015 and 2025, using the same Sept 15-
Nov 15 wet-season window as the documented 2019/2020/2022 flood events (FLOOD_EVENTS in
sar_processing.py), so that 2015/2020/2025 form a directly comparable 5-year-interval
series across the full study period. 2020 already exists (bayelsa_2020_flood_extent.tif,
one of the three documented flood years) and is reused rather than re-exported.

Important distinction from the 2019/2020/2022 maps: those three are NEMA/media-
documented flood disasters. 2015 and 2025 are NOT documented flood events -- they are
just wet-season snapshots at the start and end of the study period, classified with the
same classifier and window for direct comparability. Treat them as "what did the wet
season look like" snapshots, not "flood disaster" maps, when interpreting results.

2015 has only 4 Sentinel-1 scenes in this window (sparse, pre-dates the mission's
regular revisit pattern) vs 2025's 10 -- the 2015 composite is a weaker temporal average
and should be read with that caveat."""

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
WINDOWS = {
    "2015": ("2015-09-15", "2015-11-15"),
    "2025": ("2025-09-15", "2025-11-15"),
}

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

for year, (e_start, e_end) in WINDOWS.items():
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
    print(f"Started export task: {year} flood extent (task id: {task.id})")

print("\nDownload bayelsa_{2015,2025}_flood_extent.tif from Drive once tasks finish")
print("(same account-mismatch limitation as always).")
