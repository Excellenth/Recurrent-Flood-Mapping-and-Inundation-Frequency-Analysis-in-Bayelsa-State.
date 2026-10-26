"""Phase 4 cross-reference: NEMA/media reports named Yenagoa LGA as worst-hit in the
2022 flood. Check whether the SAR classification actually shows Yenagoa standing out
relative to Bayelsa's other 7 LGAs, as a qualitative validation against the disaster
narrative (full NEMA/SEMA record cross-referencing needs actual structured records,
which aren't available yet -- this is the check we can do without them).

The first version of this check used the raw, unmasked 2022 classification and put
coastal/riverine LGAs (Nembe, Brass) on top -- plausibly still dominated by the
swamp-forest double-bounce conflation this project spent Phase 2 correcting for, not
2022-specific flood severity. This version masks out JRC permanent water first (cheap,
static). It does NOT reapply the SAR-derived persistent-wetland mask from
build_frequency_map.py, since that requires the full 44-month computation (~1.5h) to
regenerate and isn't justified for this diagnostic alone while GEE is quota-restricted
-- so some residual swamp-forest conflation likely remains in this result too."""

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

lgas = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq("ADM1_NAME", "Bayelsa"))

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

permanent_water = (
    ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).gte(50)
)
sar_classified = sar_classified.updateMask(permanent_water.Not())

zonal = sar_classified.reduceRegions(
    collection=lgas, reducer=ee.Reducer.mean(), scale=60, tileScale=4,
)
info = zonal.getInfo()

results = []
for f in info["features"]:
    p = f["properties"]
    results.append((p.get("ADM2_NAME"), p.get("mean")))

results.sort(key=lambda x: (x[1] is None, -(x[1] or 0)))
print("LGA flood classification % (2022 event), ranked highest to lowest:")
for name, mean in results:
    pct = f"{mean*100:.1f}%" if mean is not None else "N/A"
    print(f"  {name}: {pct}")
