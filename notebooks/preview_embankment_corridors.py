"""Regenerate Figure 4: priority embankment corridors (red overlay) over wet-season
flood frequency (blue background) within the top-3 priority LGAs. Corridor definition
matches phase5_embankment_siting.py (frequency >= 0.6 AND river distance <= 500m)."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd

FREQ_THRESHOLD = 0.6
RIVER_DIST_THRESHOLD_M = 500

lgas = gpd.read_file("data/aoi/priority_lgas.geojson")
freq_cmap = mcolors.LinearSegmentedColormap.from_list("freq", ["#f7fafc", "#2b6cb0"])

fig, axes = plt.subplots(1, len(lgas), figsize=(6 * len(lgas), 6))
if len(lgas) == 1:
    axes = [axes]

for ax, (_, row) in zip(axes, lgas.iterrows()):
    name = row["ADM2_NAME"]
    geom = [row["geometry"].__geo_interface__]

    with rasterio.open("outputs/geotiffs/bayelsa_wetseason_flood_frequency_2015_2025.tif") as src:
        freq_clip, transform = mask(src, geom, crop=True, indexes=1, filled=True, nodata=np.nan)
    with rasterio.open("outputs/geotiffs/river_distance.tif") as src:
        dist_clip, _ = mask(src, geom, crop=True, indexes=1, filled=True, nodata=np.nan)

    valid = ~np.isnan(freq_clip)
    corridor = valid & (freq_clip >= FREQ_THRESHOLD) & (dist_clip <= RIVER_DIST_THRESHOLD_M)

    height, width = freq_clip.shape
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    extent = (left, right, bottom, top)

    freq_display = np.ma.masked_where(~valid, freq_clip)
    ax.imshow(freq_display, cmap=freq_cmap, vmin=0, vmax=1, extent=extent)
    corridor_display = np.ma.masked_where(~corridor, corridor)
    ax.imshow(corridor_display, cmap=mcolors.ListedColormap(["#e53e3e"]), extent=extent)

    corridor_pct = corridor[valid].mean() * 100
    ax.set_title(f"{name} (corridor {corridor_pct:.1f}% of area)")
    ax.set_xticks([])
    ax.set_yticks([])

legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor="#e53e3e", label="Priority embankment corridor"),
    plt.Rectangle((0, 0), 1, 1, facecolor="#2b6cb0", label="High wet-season flood frequency"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=2, frameon=False)
fig.suptitle("Priority embankment corridors within top-3 LGAs", fontsize=13)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig("outputs/figures/embankment_corridors_preview.png", dpi=200)
print("Saved outputs/figures/embankment_corridors_preview.png")
