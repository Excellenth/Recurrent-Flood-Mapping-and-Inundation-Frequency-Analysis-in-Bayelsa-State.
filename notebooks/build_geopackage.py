"""Consolidate all project map layers into a single GeoPackage for easy use in QGIS/
ArcGIS: AOI + LGA boundaries (vector) and every raster map product (flood-extent,
frequency, river distance, population).

GDAL's GeoPackage raster driver only supports single-band raster tables for non-Byte
pixel types, so the 4-band frequency map is split into 4 separate raster tables rather
than kept as one multi-band layer -- a GPKG-native constraint, not a data loss.

PROJ_LIB/PROJ_DATA are pinned to rasterio's own bundled PROJ database before anything
else runs: this machine has a system-wide PROJ_LIB pointing at a PostGIS-bundled proj.db
of an incompatible version, which otherwise breaks GPKG writes (CRS handling) with a
cryptic PROJ error while leaving plain GeoTIFF reads unaffected."""

import os

_rasterio_proj = os.path.join(os.path.dirname(__import__("rasterio").__file__), "proj_data")
os.environ["PROJ_LIB"] = _rasterio_proj
os.environ["PROJ_DATA"] = _rasterio_proj

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.shutil import copy as rio_copy

GPKG_PATH = "outputs/Bayelsa_Flood_SAR.gpkg"
if os.path.exists(GPKG_PATH):
    os.remove(GPKG_PATH)

# GDAL's GPKG "2d gridded coverage" extension (used for non-Byte single-band rasters)
# cannot record NaN as a nodata value -- it silently drops the nodata tag instead of
# erroring, so masked pixels read back as literal 0.0 rather than nodata unless a real
# numeric sentinel is substituted first. Byte/tile rasters (flood_extent, no NaN) are
# unaffected and don't need this.
NODATA_SENTINEL = -9999.0


def add_raster_band(src_path, table_name, band=1):
    with rasterio.open(src_path) as src:
        data = src.read(band)
        profile = src.profile.copy()
        profile.update(count=1)
        if profile.get("dtype", "").startswith("float") and np.isnan(data).any():
            data = np.where(np.isnan(data), NODATA_SENTINEL, data)
            profile["nodata"] = NODATA_SENTINEL

        band_path = f"/vsimem/{table_name}.tif"
        with rasterio.open(band_path, "w", **profile) as dst:
            dst.write(data, 1)
            if src.descriptions and len(src.descriptions) >= band and src.descriptions[band - 1]:
                dst.set_band_description(1, src.descriptions[band - 1])
        rio_copy(band_path, GPKG_PATH, driver="GPKG", RASTER_TABLE=table_name,
                  APPEND_SUBDATASET="YES", strict=False)
    print(f"  raster layer: {table_name}")


# --- Vector layers ---
print("Vector layers:")
aoi = gpd.read_file("data/aoi/study area.shp")
aoi.to_file(GPKG_PATH, layer="aoi_boundary", driver="GPKG")
print("  vector layer: aoi_boundary")

lgas_all = gpd.read_file("data/aoi/bayelsa_lgas_all.geojson")
lgas_all.to_file(GPKG_PATH, layer="lgas_all_8", driver="GPKG")
print("  vector layer: lgas_all_8")

priority_lgas = gpd.read_file("data/aoi/priority_lgas.geojson")
priority_lgas.to_file(GPKG_PATH, layer="priority_lgas_top3", driver="GPKG")
print("  vector layer: priority_lgas_top3")

lgas_lstm_risk = gpd.read_file("data/aoi/bayelsa_lgas_lstm_risk.geojson")
lgas_lstm_risk.to_file(GPKG_PATH, layer="lgas_lstm_risk_index", driver="GPKG")
print("  vector layer: lgas_lstm_risk_index")

# --- Raster layers ---
print("Raster layers:")
for year in ["2016", "2019", "2020", "2022", "2025"]:
    add_raster_band(f"outputs/geotiffs/bayelsa_{year}_flood_extent.tif", f"flood_extent_{year}")

freq_bands = ["flood_frequency", "n_valid_months", "permanent_water", "sar_persistent_wetland"]
for i, name in enumerate(freq_bands, start=1):
    add_raster_band("outputs/geotiffs/bayelsa_wetseason_flood_frequency_2015_2025.tif",
                     f"frequency_{name}", band=i)

add_raster_band("outputs/geotiffs/river_distance.tif", "river_distance")
add_raster_band("outputs/geotiffs/bayelsa_worldpop_2020.tif", "worldpop_2020")
add_raster_band("outputs/geotiffs/bayelsa_pixel_forecast_2026_2030.tif", "pixel_forecast_2026_2030")

size_mb = os.path.getsize(GPKG_PATH) / 1e6
print(f"\nSaved {GPKG_PATH} ({size_mb:.1f} MB)")
