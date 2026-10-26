"""Generate a 3-panel comparison preview of the classified flood-extent maps
(2019, 2020, 2022) produced by export_flood_extent_maps.py -- the first time these
three events have been visually comparable side by side, rather than only the 2022
event (Figure 2) or the 11-year frequency aggregate (Figure 1)."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np

aoi = gpd.read_file("data/aoi/study area.shp")
geom = [aoi.geometry.union_all().__geo_interface__]

YEARS = ["2019", "2020", "2022"]
cmap = mcolors.ListedColormap(["#2b6cb0", "#c53030"])  # 0=non-flood (blue), 1=flood (red)

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
    ax.set_title(f"{year} ({pct_flooded:.1f}% flooded)")
    ax.set_xticks([])
    ax.set_yticks([])

legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor="#c53030", label="Classified flood"),
    plt.Rectangle((0, 0), 1, 1, facecolor="#2b6cb0", label="Classified non-flood"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=2, frameon=False)
fig.suptitle("Classified flood extent by event year, Bayelsa State", fontsize=13)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig("outputs/figures/flood_extent_comparison_2019_2020_2022.png", dpi=200)
print("Saved outputs/figures/flood_extent_comparison_2019_2020_2022.png")
