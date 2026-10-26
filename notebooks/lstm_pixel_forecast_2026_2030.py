"""Per-pixel 2026-2030 flood-risk forecast, produced under a same-day submission
deadline as a documented scope decision -- see PROJECT_PLAN.md Phase 3.

Why this approach and not a genuine per-pixel LSTM: forecasting 2026-2030 per pixel
"from scratch" would mean training a separate time-series model per pixel (or a much
larger joint model) over ~1.69M pixels, each needing its own multi-year monthly
lookback -- the exact compute cost that forced the LSTM to zonal (8-LGA) granularity
in Phase 3 in the first place (GEE quota was exhausted doing far less than this).
Redoing that at pixel scale under time pressure risks an unvalidated model or a fresh
quota exhaustion with no time to recover.

Instead: statistical downscaling (delta method, standard in climate/hydrology for
combining a coarse temporal forecast with a fine-resolution spatial climatology).
The already-validated LGA-level LSTM risk index supplies the *temporal* forecast
signal (how each zone's risk is projected to change); the already-validated pixel-
level wet-season flood-frequency map supplies the *spatial* pattern (which pixels
within a zone are chronically flood-prone -- river corridors and swamp forest vs.
upland). Combining them, via an ADDITIVE delta (not a multiplicative ratio -- tried first,
rejected: flood_frequency is a bounded [0,1] proportion with mean already 0.515, and
multiplying by the ~1.3-2.0x zonal scale factors pushed 35.7% of all pixels past 1.0,
i.e. straight into the clip ceiling -- exactly the already flood-prone areas losing all
spatial differentiation, the opposite of what a spatial map is for. Multiplicative
ratios are the right delta-method choice for unbounded, always-positive quantities
like rainfall; for a bounded proportion, the standard choice is additive):

    pixel_forecast[p] = flood_frequency[p] + (risk_index_baseline_2026_2030[z(p)]
                                               - risk_index_test_2024_2025[z(p)])

i.e. each pixel's known historical flood-frequency is shifted by its LGA's projected
absolute change in risk index between the most recent observed period (2024-2025,
real data) and the 2026-2030 baseline (flat "no-trend" projection, see Phase 3).
Result clipped to [0, 1] since it is still a frequency-like quantity.

Limitation, stated plainly: this assumes the *spatial pattern* of flood risk within
each LGA stays fixed and only its overall *level* shifts by the zonal factor -- it
cannot capture a scenario where the flood-prone corridor itself relocates. The
2026-2030 zonal baseline is also flat by construction (Phase 3), so this pixel map
inherits that same "no further climate/land-use trend" assumption. It is a spatial
disaggregation of an already-documented zonal forecast, not an independent pixel-
level prediction -- this must be stated as such in the paper, not presented as
equivalent in kind to the real-data classified maps (2016/2019/2020/2022/2025)."""

import os

_rasterio_proj = os.path.join(os.path.dirname(__import__("rasterio").__file__), "proj_data")
os.environ["PROJ_LIB"] = _rasterio_proj
os.environ["PROJ_DATA"] = _rasterio_proj

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

FREQ_PATH = "outputs/geotiffs/bayelsa_wetseason_flood_frequency_2015_2025.tif"
LGA_PATH = "data/aoi/bayelsa_lgas_lstm_risk.geojson"
OUT_PATH = "outputs/geotiffs/bayelsa_pixel_forecast_2026_2030.tif"

with rasterio.open(FREQ_PATH) as src:
    freq = src.read(1, masked=True).astype("float64")
    profile = src.profile.copy()
    transform = src.transform
    shape = src.shape

lgas = gpd.read_file(LGA_PATH)
assert lgas.crs.to_epsg() == 4326

lgas["delta"] = (
    lgas["risk_index_baseline_2026_2030"] - lgas["risk_index_test_2024_2025"]
)
print(lgas[["ADM2_NAME", "risk_index_test_2024_2025",
            "risk_index_baseline_2026_2030", "delta"]])

zone_id_raster = rasterize(
    [(geom, i) for i, geom in enumerate(lgas.geometry, start=1)],
    out_shape=shape,
    transform=transform,
    fill=0,
    dtype="int32",
)

delta_lookup = np.zeros(len(lgas) + 1, dtype="float64")
for i, delta in enumerate(lgas["delta"], start=1):
    delta_lookup[i] = delta
delta_raster = delta_lookup[zone_id_raster]

pixel_forecast = np.where(
    zone_id_raster > 0,
    np.clip(freq.filled(np.nan) + delta_raster, 0.0, 1.0),
    np.nan,
)

profile.update(count=1, dtype="float32", nodata=-9999.0)
out_data = np.where(np.isnan(pixel_forecast), -9999.0, pixel_forecast).astype("float32")

with rasterio.open(OUT_PATH, "w", **profile) as dst:
    dst.write(out_data, 1)
    dst.set_band_description(1, "pixel_forecast_risk_2026_2030")

valid = pixel_forecast[~np.isnan(pixel_forecast)]
print(f"\nSaved {OUT_PATH}")
print(f"Valid pixels: {valid.size:,} | mean {valid.mean():.3f} | "
      f"median {np.median(valid):.3f} | max {valid.max():.3f}")
