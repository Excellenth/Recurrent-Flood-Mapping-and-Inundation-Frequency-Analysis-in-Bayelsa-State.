"""Zonal (LGA-level) monthly time series for LSTM flood-risk prediction (Objective 3).

Deliberately lightweight compared to Phase 2's per-pixel classification: no Refined Lee
speckle filter, no GLCM texture, no per-pixel RF classification. Zonal averaging over
each LGA's many pixels already suppresses speckle on its own, so the expensive
per-pixel filtering isn't needed for a zone-level time series. This trades per-pixel
classified flood extent for a coarser zone-level continuous SAR/rainfall signal -- a
deliberate scoping choice to keep this tractable under the GEE compute quota
restriction, not a hidden shortcut.
"""

import ee

from sar_processing import get_orbit_collection, get_daily_mosaics


def bayelsa_lgas():
    """Lazily constructed -- must not run at import time, before init_ee()."""
    return ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq("ADM1_NAME", "Bayelsa"))


def monthly_raw_composite(aoi, year, month):
    """Median VV/VH composite for one month, no despeckling/texture -- cheap.

    Some months (early in the Sentinel-1 record, before its revisit pattern
    stabilized around 2015-2016) have zero scenes for this orbit. median() on an
    empty collection returns a zero-band image, and .select(['VV','VH']) on that
    hard-errors rather than just being masked -- so check first and fall back to a
    properly-masked placeholder with the right band names for genuinely empty months."""
    start = f"{year}-{month:02d}-01"
    end = ee.Date(start).advance(1, "month")
    coll = get_orbit_collection(aoi, start=start, end=end)
    daily = get_daily_mosaics(coll, aoi)

    placeholder = ee.Image.constant([0, 0]).rename(["VV", "VH"]).updateMask(ee.Image.constant(0))
    composite = ee.Image(
        ee.Algorithms.If(daily.size().gt(0), daily.median().select(["VV", "VH"]), placeholder)
    )
    return composite


def monthly_rainfall(aoi, year, month):
    start = f"{year}-{month:02d}-01"
    end = ee.Date(start).advance(1, "month")
    chirps = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(start, end)
        .filterBounds(aoi)
        .sum()
        .rename("rainfall_mm")
    )
    return chirps


def zonal_month_features(aoi, year, month, lgas=None):
    """Zonal mean VV/VH + total rainfall per LGA for one month.
    Returns an ee.FeatureCollection with one feature per LGA."""
    lgas = lgas if lgas is not None else bayelsa_lgas()
    sar = monthly_raw_composite(aoi, year, month)
    rain = monthly_rainfall(aoi, year, month)
    combined = sar.addBands(rain)

    zonal = combined.reduceRegions(collection=lgas, reducer=ee.Reducer.mean(), scale=60, tileScale=4)

    def tag(f):
        return f.set({"year": year, "month": month})

    return zonal.map(tag)
