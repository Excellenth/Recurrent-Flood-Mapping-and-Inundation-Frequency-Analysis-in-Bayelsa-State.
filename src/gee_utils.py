"""Shared GEE initialization and AOI loading for the Bayelsa flood SAR project."""

from pathlib import Path

import ee
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AOI_SHAPEFILE = PROJECT_ROOT / "data" / "aoi" / "study area.shp"


def init_ee():
    """Initialize the Earth Engine API using cached credentials."""
    ee.Initialize()


def load_aoi_gdf():
    """Load the Bayelsa AOI as a GeoDataFrame (EPSG:4326)."""
    return gpd.read_file(AOI_SHAPEFILE)


def load_aoi_ee():
    """Load the Bayelsa AOI as an ee.Geometry via its GeoJSON representation."""
    gdf = load_aoi_gdf()
    geom = gdf.geometry.union_all()
    return ee.Geometry(geom.__geo_interface__)
