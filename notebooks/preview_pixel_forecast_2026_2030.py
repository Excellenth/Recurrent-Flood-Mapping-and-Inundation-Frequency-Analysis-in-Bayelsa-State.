"""Preview figure for the per-pixel 2026-2030 flood-risk forecast
(lstm_pixel_forecast_2026_2030.py) alongside the current pixel-level flood-frequency
map, so the spatial change this downscaling projects is visible directly, not just
implied by the LGA-level delta table."""

import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np

aoi = gpd.read_file("data/aoi/study area.shp")
geom = [aoi.geometry.union_all().__geo_interface__]

PANELS = [
    ("outputs/geotiffs/bayelsa_wetseason_flood_frequency_2015_2025.tif", 1,
     "Observed wet-season flood frequency\n(2015-2025 mean)"),
    ("outputs/geotiffs/bayelsa_pixel_forecast_2026_2030.tif", 1,
     "Projected 2026-2030 flood-risk forecast\n(downscaled from LGA-level LSTM)"),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 6))

for ax, (path, band, title) in zip(axes, PANELS):
    with rasterio.open(path) as src:
        clipped, transform = mask(src, geom, crop=True, indexes=band, filled=True, nodata=-9999.0)
        arr = clipped
        valid = arr != -9999.0
        display = np.ma.masked_where(~valid, arr)
        height, width = arr.shape
        left, top = transform * (0, 0)
        right, bottom = transform * (width, height)
        extent = (left, right, bottom, top)
        mean_val = arr[valid].mean()

    im = ax.imshow(display, cmap="YlOrRd", vmin=0, vmax=1, extent=extent)
    aoi.boundary.plot(ax=ax, color="black", linewidth=0.5)
    ax.set_title(f"{title}\n(mean {mean_val:.2f})")
    ax.set_xticks([])
    ax.set_yticks([])

fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.05,
             label="Flood frequency / risk index (0-1)")
fig.suptitle("Bayelsa State: observed flood frequency vs. 2026-2030 pixel-level forecast", fontsize=12)
fig.savefig("outputs/figures/pixel_forecast_2026_2030_preview.png", dpi=200, bbox_inches="tight")
print("Saved outputs/figures/pixel_forecast_2026_2030_preview.png")
