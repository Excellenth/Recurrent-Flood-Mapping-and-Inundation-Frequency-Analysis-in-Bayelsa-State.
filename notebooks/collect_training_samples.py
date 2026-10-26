"""Collect RF training samples for each flood-event year separately (each run is near
GEE's ~5min interactive-compute ceiling) and save them locally as JSON. Keeping this as
a standalone, resumable step means the expensive part isn't lost if a later step fails."""

import sys
import json
import time

sys.path.insert(0, "src")

import ee
from gee_utils import init_ee, load_aoi_ee
from classification import build_event_stack_and_labels

init_ee()
aoi = load_aoi_ee()
dem = ee.Image("MERIT/DEM/v1_0_3")

POINTS_PER_CLASS = 300
SCALE = 60  # coarser than the 20m tried earlier -- stratifiedSample/sampleRegions cost
            # scales with candidate pixel count (region_area / scale^2), and this was
            # timing out at 20m. Despeckled SAR is already effectively smoother than
            # 20m from the 7x7 Refined Lee window, so 60m sampling is well justified.
MAX_ATTEMPTS = 3

for year in ["2019", "2020", "2022"]:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        t0 = time.time()
        try:
            _, samples = build_event_stack_and_labels(aoi, dem, year, points_per_class=POINTS_PER_CLASS, scale=SCALE)
            info = samples.getInfo()
            records = [f["properties"] for f in info["features"]]
            out_path = f"data/training_samples_{year}.json"
            with open(out_path, "w") as fh:
                json.dump(records, fh)
            print(f"{year}: {len(records)} samples, took {round(time.time()-t0,1)}s -> {out_path}")
            break
        except Exception as e:
            print(f"{year}: attempt {attempt} failed after {round(time.time()-t0,1)}s: {e}")
            if attempt == MAX_ATTEMPTS:
                raise

print("DONE")
