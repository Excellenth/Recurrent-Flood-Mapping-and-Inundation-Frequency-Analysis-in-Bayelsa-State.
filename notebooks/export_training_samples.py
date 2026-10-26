"""One-off script: build training samples for each flood-event year and export them
as GEE table assets (async batch task -- avoids the interactive-compute timeout that
stratifiedSample hits when run synchronously over the full AOI's GLCM/feature stack)."""

import sys
sys.path.insert(0, "src")

import ee
from gee_utils import init_ee, load_aoi_ee
from classification import build_event_stack_and_labels

init_ee()
aoi = load_aoi_ee()
dem = ee.Image("MERIT/DEM/v1_0_3")

PROJECT = "ee-seunlekan"
POINTS_PER_CLASS = 500
SAMPLE_SCALE = 30

for year in ["2019", "2020", "2022"]:
    _, samples = build_event_stack_and_labels(aoi, dem, year, points_per_class=POINTS_PER_CLASS)
    asset_id = f"projects/{PROJECT}/assets/bayelsa_train_{year}"
    task = ee.batch.Export.table.toAsset(
        collection=samples,
        description=f"bayelsa_train_{year}",
        assetId=asset_id,
    )
    task.start()
    print(f"Started table export: {asset_id} (task id: {task.id})")

print("\nCheck status with earthengine task list, or https://code.earthengine.google.com/tasks")
