"""One-off script: build despeckled event/baseline SAR composites for 2019, 2020, 2022
and export them to Google Drive as GeoTIFFs (VV, VH bands, dB)."""

import sys
sys.path.insert(0, "src")

import ee
from gee_utils import init_ee, load_aoi_ee
from sar_processing import (
    get_orbit_collection, get_daily_mosaics, despeckle_collection,
    event_composite, baseline_composite, FLOOD_EVENTS, DRY_SEASON_BASELINES,
)

init_ee()
aoi = load_aoi_ee()

EXPORT_SCALE = 20  # meters
DRIVE_FOLDER = "BayelsaFloodSAR"

tasks = []
for year in ["2019", "2020", "2022"]:
    e_start, e_end = FLOOD_EVENTS[year]
    b_start, b_end = DRY_SEASON_BASELINES[year]
    # pull enough of the year to cover both windows in one collection
    coll = get_orbit_collection(aoi, start=f"{year}-01-01", end=f"{year}-12-01")
    daily = get_daily_mosaics(coll, aoi)
    despeckled = despeckle_collection(daily)

    event_img = event_composite(despeckled, year).select(["VV", "VH"])
    base_img = baseline_composite(despeckled, year).select(["VV", "VH"])

    for label, img in [(f"{year}_flood_event", event_img), (f"{year}_dry_baseline", base_img)]:
        task = ee.batch.Export.image.toDrive(
            image=img.toFloat(),
            description=f"bayelsa_{label}",
            folder=DRIVE_FOLDER,
            fileNamePrefix=f"bayelsa_{label}",
            region=aoi,
            scale=EXPORT_SCALE,
            crs="EPSG:4326",
            maxPixels=1e9,
        )
        task.start()
        tasks.append((label, task))
        print(f"Started export task: {label} (task id: {task.id})")

print("\nAll 6 export tasks started. Check status with earthengine task list, or:")
print("  https://code.earthengine.google.com/tasks")
