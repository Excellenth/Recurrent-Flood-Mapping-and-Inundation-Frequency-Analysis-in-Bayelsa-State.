"""Phase 4 follow-up, fully local: rerun the LGA-level 2022 flood-classification ranking
with BOTH exclusion layers applied (JRC permanent water + SAR-derived persistent
wetland), not just JRC as in lga_zonal_check.py. That script's docstring explicitly
flagged this gap -- it only masked JRC because reapplying the SAR-persistent-wetland
mask required the full 44-month computation, and GEE was quota-restricted at the time.

That computation has already been run once for build_frequency_map.py and downloaded
locally; its sar_persistent_wetland band is a static per-pixel mask, pixel-aligned with
bayelsa_2022_flood_extent.tif (same grid/transform/CRS, confirmed directly), so both
exclusions can be applied here with zero new GEE compute."""

import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd

FREQ_PATH = "outputs/geotiffs/bayelsa_wetseason_flood_frequency_2015_2025.tif"
EXTENT_PATH = "outputs/geotiffs/bayelsa_2022_flood_extent.tif"
LGA_PATH = "data/aoi/bayelsa_lgas_all.geojson"

lgas = gpd.read_file(LGA_PATH)

with rasterio.open(EXTENT_PATH) as src:
    pixel_size_m = abs(src.transform.a) * 111320
    pixel_area_km2 = (pixel_size_m ** 2) / 1e6

results = []
for _, row in lgas.iterrows():
    name = row["ADM2_NAME"]
    geom = [row["geometry"].__geo_interface__]

    with rasterio.open(EXTENT_PATH) as src:
        extent_clip, _ = mask(src, geom, crop=True, indexes=1, filled=True, nodata=255)
    with rasterio.open(FREQ_PATH) as src:
        freq_clip = mask(src, geom, crop=True, filled=True, nodata=np.nan)[0]
        permanent_water = freq_clip[2]
        sar_persistent_wetland = freq_clip[3]

    in_lga = extent_clip != 255
    excluded = (permanent_water == 1) | (sar_persistent_wetland == 1)
    valid = in_lga & ~excluded

    n_valid = valid.sum()
    n_flood = (valid & (extent_clip == 1)).sum()
    pct_flood_masked = n_flood / n_valid * 100 if n_valid else None

    n_valid_jrc_only = (in_lga & (permanent_water != 1)).sum()
    n_flood_jrc_only = (in_lga & (permanent_water != 1) & (extent_clip == 1)).sum()
    pct_flood_jrc_only = n_flood_jrc_only / n_valid_jrc_only * 100 if n_valid_jrc_only else None

    results.append({
        "LGA": name,
        "pct_flood_jrc_only": round(pct_flood_jrc_only, 1) if pct_flood_jrc_only is not None else None,
        "pct_flood_dual_masked": round(pct_flood_masked, 1) if pct_flood_masked is not None else None,
        "lga_area_km2": round(in_lga.sum() * pixel_area_km2, 1),
        "excluded_pct_of_lga": round(excluded[in_lga].sum() / in_lga.sum() * 100, 1) if in_lga.sum() else None,
    })

import pandas as pd
df = pd.DataFrame(results).sort_values("pct_flood_dual_masked", ascending=False)
df.to_csv("outputs/tables/lga_zonal_check_2022_masked.csv", index=False)
print("LGA flood classification %, 2022 event, JRC-only vs. JRC+SAR-persistent-wetland masked:")
print(df.to_string(index=False))
print("\nSaved to outputs/tables/lga_zonal_check_2022_masked.csv")
