"""Phase 5, pixel-level population weighting (open item from PROJECT_PLAN.md): rank
individual embankment-corridor segments within the top-3 priority LGAs by estimated
population protected, using the pixel-level WorldPop 2020 raster that was stuck queued
under GEE's quota-restricted mode until now -- answering "which specific corridor
segments protect the most people," not just which LGA to prioritize (phase5_priority_
ranking.py) or how much corridor a whole LGA has (phase5_embankment_siting.py).

Protection catchment: each segment's corridor pixels are vectorized and buffered by
PROTECTION_BUFFER_M (1km, a rough proxy for the area a community-scale embankment
segment would shield from a river-corridor flood event -- not a hydrological/engineering
model), and population within that buffer is summed directly from the native-resolution
WorldPop raster via vector masking. This avoids resampling population counts onto the
coarser 60m corridor grid, which would require area-correcting the counts to avoid
double- or under-counting. Buffer distance is converted degrees<->meters with a flat
111,320 m/degree approximation (fine at this AOI's latitude and buffer scale)."""

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, MultiPolygon
from scipy import ndimage

FREQ_THRESHOLD = 0.6
RIVER_DIST_THRESHOLD_M = 500
MIN_SEGMENT_PIXELS = 5
PROTECTION_BUFFER_M = 1000
BUFFER_DEG = PROTECTION_BUFFER_M / 111320

lgas = gpd.read_file("data/aoi/priority_lgas.geojson")
pop_src = rasterio.open("outputs/geotiffs/bayelsa_worldpop_2020.tif")

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

    for seg_id, size in enumerate(sizes, start=1):
        if size < MIN_SEGMENT_PIXELS:
            continue
        seg_mask = (labeled == seg_id)

        polys = [
            shape(g) for g, v in shapes(seg_mask.astype("uint8"), mask=seg_mask, transform=clip_transform)
            if v == 1
        ]
        if not polys:
            continue
        segment_geom = MultiPolygon(polys) if len(polys) > 1 else polys[0]
        catchment = segment_geom.buffer(BUFFER_DEG)

        try:
            pop_clip, _ = mask(pop_src, [catchment.__geo_interface__], crop=True, filled=True, nodata=0)
            pop_sum = float(np.nansum(pop_clip))
        except ValueError:
            pop_sum = 0.0

        rows, cols = np.where(seg_mask)
        lon, lat = rasterio.transform.xy(clip_transform, rows.mean(), cols.mean())

        results.append({
            "LGA": name,
            "segment_id": seg_id,
            "segment_pixels": int(size),
            "centroid_lon": round(lon, 4),
            "centroid_lat": round(lat, 4),
            "est_population_protected": round(pop_sum, 0),
        })

pop_src.close()

df = pd.DataFrame(results).sort_values("est_population_protected", ascending=False)
df.to_csv("outputs/tables/phase5_corridor_population_weighting.csv", index=False)

print(f"{len(df)} significant corridor segments (>={MIN_SEGMENT_PIXELS}px) across top-3 LGAs.\n")
print("Top 15 segments by estimated population protected (1km catchment buffer):")
print(df.head(15).to_string(index=False))
print("\nTop segment per LGA:")
print(df.loc[df.groupby("LGA")["est_population_protected"].idxmax()].to_string(index=False))
print("\nSaved to outputs/tables/phase5_corridor_population_weighting.csv")
