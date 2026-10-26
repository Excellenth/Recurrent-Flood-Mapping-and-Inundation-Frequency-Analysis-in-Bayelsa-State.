"""Objective 5, pixel-level pass: identify priority flood-break embankment corridors
within the top-3 priority LGAs (Southern Ijaw, Brass, Yenegoa per the LGA-level
ranking), using the corrected frequency map + locally-derived river distance.

Corridor definition: high chronic flood frequency (>=0.6, roughly the top quartile of
the statewide distribution) AND close to a river/permanent-water channel (<=500m) --
i.e. land that floods often AND sits right where an embankment would actually need to
go to protect it. Population weighting (WorldPop) is not yet included -- that export
got stuck queued under the GEE quota-restricted mode; this identifies WHERE the
hydrology says embankments matter, population will refine WHICH segments to
prioritize once available.
"""

import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from scipy import ndimage

FREQ_THRESHOLD = 0.6
RIVER_DIST_THRESHOLD_M = 500

lgas = gpd.read_file("data/aoi/priority_lgas.geojson")

with rasterio.open("outputs/geotiffs/bayelsa_wetseason_flood_frequency_2015_2025.tif") as freq_src:
    freq_full = freq_src.read(1)
    freq_transform = freq_src.transform
    freq_crs = freq_src.crs

with rasterio.open("outputs/geotiffs/river_distance.tif") as dist_src:
    dist_full = dist_src.read(1)

pixel_size_m = abs(freq_transform.a) * 111320
pixel_area_km2 = (pixel_size_m ** 2) / 1e6

results = []
for _, lga_row in lgas.iterrows():
    name = lga_row["ADM2_NAME"]
    geom = [lga_row["geometry"].__geo_interface__]

    with rasterio.open("outputs/geotiffs/bayelsa_wetseason_flood_frequency_2015_2025.tif") as freq_src:
        freq_clip, clip_transform = mask(freq_src, geom, crop=True, indexes=1, filled=True, nodata=np.nan)
    with rasterio.open("outputs/geotiffs/river_distance.tif") as dist_src:
        dist_clip, _ = mask(dist_src, geom, crop=True, indexes=1, filled=True, nodata=np.nan)

    valid = ~np.isnan(freq_clip)
    corridor = valid & (freq_clip >= FREQ_THRESHOLD) & (dist_clip <= RIVER_DIST_THRESHOLD_M)

    labeled, n_segments = ndimage.label(corridor)
    sizes = ndimage.sum(corridor, labeled, range(1, n_segments + 1))
    # Keep only segments with a meaningful footprint (>= ~5 pixels ~ 0.018 km2) to
    # drop single-pixel noise.
    significant = (sizes >= 5).sum()

    corridor_area_km2 = corridor.sum() * pixel_area_km2
    lga_area_km2 = valid.sum() * pixel_area_km2

    # Centroid of the largest segment, in the clipped raster's own coordinates ->
    # convert to lon/lat for reporting.
    if n_segments > 0:
        largest_label = np.argmax(sizes) + 1
        rows, cols = np.where(labeled == largest_label)
        r_c, c_c = rows.mean(), cols.mean()
        lon, lat = rasterio.transform.xy(clip_transform, r_c, c_c)
    else:
        lon, lat = None, None

    results.append({
        "LGA": name,
        "lga_area_km2": round(lga_area_km2, 1),
        "corridor_area_km2": round(corridor_area_km2, 2),
        "corridor_pct_of_lga": round(corridor_area_km2 / lga_area_km2 * 100, 2) if lga_area_km2 else None,
        "n_corridor_segments": int(n_segments),
        "n_significant_segments": int(significant),
        "largest_segment_centroid_lon": round(lon, 4) if lon else None,
        "largest_segment_centroid_lat": round(lat, 4) if lat else None,
    })

import pandas as pd
df = pd.DataFrame(results)
df.to_csv("outputs/tables/phase5_embankment_corridors.csv", index=False)
print(df.to_string(index=False))
print("\nSaved to outputs/tables/phase5_embankment_corridors.csv")
