"""Pull a few true-color Sentinel-2 thumbnails at locations where S2 says 'dry'
but the SAR classifier says 'flood', to see whether these look like forest canopy
(supports the flooded-vegetation/double-bounce explanation) or bare/dry land
(would suggest genuine SAR over-classification)."""

import sys
import json
import urllib.request

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
s2_water = ndwi.gt(0.0).rename("s2_water")

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

disagreement = s2_water.eq(0).And(sar_classified.eq(1)).rename("disagree")
combined = disagreement.addBands(ee.Image.pixelLonLat())

pts = combined.updateMask(disagreement).stratifiedSample(
    numPoints=5, classBand="disagree", region=aoi, scale=20, seed=7, geometries=True, tileScale=4,
)
pts_info = pts.getInfo()
print("Disagreement sample points:", len(pts_info["features"]))

vis = s2_mosaic.visualize(bands=["B4", "B3", "B2"], min=0, max=3000)

for i, f in enumerate(pts_info["features"]):
    coords = f["geometry"]["coordinates"]
    pt = ee.Geometry.Point(coords)
    region = pt.buffer(600).bounds()
    url = vis.getThumbURL({"region": region, "dimensions": 400, "format": "png"})
    out_path = f"outputs/figures/disagreement_point_{i}_{coords[0]:.4f}_{coords[1]:.4f}.png"
    urllib.request.urlretrieve(url, out_path)
    print(f"Point {i}: lon={coords[0]:.5f}, lat={coords[1]:.5f} -> {out_path}")
