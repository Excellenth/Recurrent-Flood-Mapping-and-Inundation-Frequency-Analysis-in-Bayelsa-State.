"""3-panel comparison of classified flood extent at roughly 5-year intervals across the
study period: 2016, 2020, 2025 (2015 substituted with 2016 -- see
export_2016_flood_extent.py and project memory for why: 2015's composite showed a
systematic backscatter offset from the training-era baseline that persisted even after
widening the compositing window, most likely an early-Sentinel-1-mission calibration
difference, and gave an implausible ~80% flooded estimate).

Unlike the 2019/2020/2022 documented-flood-event comparison (Figure 2b), these three
are wet-season SNAPSHOTS at regular intervals, not all NEMA/media-documented disasters --
only 2020 is a documented event. Labelled accordingly."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np

aoi = gpd.read_file("data/aoi/study area.shp")
geom = [aoi.geometry.union_all().__geo_interface__]

YEARS = ["2016", "2020", "2025"]
LABELS = {"2016": "2016 (substitute for 2015)", "2020": "2020 (documented event)", "2025": "2025"}
cmap = mcolors.ListedColormap(["#2b6cb0", "#c53030"])

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

for ax, year in zip(axes, YEARS):
    path = f"outputs/geotiffs/bayelsa_{year}_flood_extent.tif"
    with rasterio.open(path) as src:
        clipped, transform = mask(src, geom, crop=True, filled=True, nodata=255)
        band = clipped[0]
        valid = band != 255
        pct_flooded = (band[valid] == 1).mean() * 100

        display = np.ma.masked_where(~valid, band)
        height, width = band.shape
        left, top = transform * (0, 0)
        right, bottom = transform * (width, height)
        extent = (left, right, bottom, top)

    ax.imshow(display, cmap=cmap, vmin=0, vmax=1, extent=extent)
    aoi.boundary.plot(ax=ax, color="black", linewidth=0.5)
    ax.set_title(f"{LABELS[year]}\n({pct_flooded:.1f}% flooded)")
    ax.set_xticks([])
    ax.set_yticks([])

legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor="#c53030", label="Classified flood"),
    plt.Rectangle((0, 0), 1, 1, facecolor="#2b6cb0", label="Classified non-flood"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=2, frameon=False)
fig.suptitle("Classified flood extent at ~5-year intervals, Bayelsa State (Sept15-Nov15 wet season)", fontsize=12)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig("outputs/figures/flood_extent_5yr_interval_2016_2020_2025.png", dpi=200)
print("Saved outputs/figures/flood_extent_5yr_interval_2016_2020_2025.png")
