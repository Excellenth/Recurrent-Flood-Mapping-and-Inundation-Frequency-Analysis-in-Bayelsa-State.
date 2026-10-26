"""Sentinel-1 SAR preprocessing for the Bayelsa flood mapping project.

Primary track: ASCENDING relative orbit 103, which gives near-full single-scene
coverage of the Bayelsa AOI and mosaics to 100% coverage on multi-tile dates.
See notebooks/01_sar_processing.ipynb for the coverage analysis behind this choice.
"""

import datetime

import ee

PRIMARY_ORBIT_PASS = "ASCENDING"
PRIMARY_RELATIVE_ORBIT = 103

# Flood-event windows anchored on documented NEMA/SEMA/media-reported peaks.
# 2019: flooding began June, peaked late Sept, continued to Nov (~700k displaced in Bayelsa).
# 2020: began Sept, worst flooding early Oct (smaller local impact than 2019/2022).
# 2022: Sept-Nov, peaked Oct-Nov, most severe (1.3M displaced in Bayelsa) - best documented.
# Sources: reliefweb.int/report/nigeria/acaps-briefing-note-nigeria-floods-borno-delta-kebbi-and-kogi-states-17-october-2019,
# floodlist.com/africa/nigeria-floods-october-2020, en.wikipedia.org/wiki/2022_Bayelsa_State_floods,
# nema.gov.ng/nema-assesses-flood-disaster-in-bayelsa-state/
FLOOD_EVENTS = {
    "2019": ("2019-09-15", "2019-11-15"),
    "2020": ("2020-09-15", "2020-10-31"),
    "2022": ("2022-09-15", "2022-11-15"),
}

# Dry-season baseline windows (same years) for non-flood training/contrast samples.
DRY_SEASON_BASELINES = {
    "2019": ("2019-01-15", "2019-03-15"),
    "2020": ("2020-01-15", "2020-03-15"),
    "2022": ("2022-01-15", "2022-03-15"),
}


def get_orbit_collection(aoi, pass_dir=PRIMARY_ORBIT_PASS, relative_orbit=PRIMARY_RELATIVE_ORBIT,
                          start=None, end=None):
    """Sentinel-1 GRD IW VV+VH collection filtered to a single consistent-geometry track."""
    coll = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("orbitProperties_pass", pass_dir))
        .filter(ee.Filter.eq("relativeOrbitNumber_start", relative_orbit))
    )
    if start is not None and end is not None:
        coll = coll.filterDate(start, end)
    return coll


def get_daily_mosaics(collection, aoi):
    """Mosaic same-date tiles (needed because the AOI spans two along-track frames)
    and tag each output image with its acquisition date."""
    dates = collection.aggregate_array("system:time_start").map(
        lambda t: ee.Date(t).format("YYYY-MM-dd")
    ).distinct()

    def build_day(date_str):
        date_str = ee.String(date_str)
        day_start = ee.Date(date_str)
        day_end = day_start.advance(1, "day")
        day_coll = collection.filterDate(day_start, day_end)
        mosaic = day_coll.mosaic().clip(aoi)
        return mosaic.set({
            "date": date_str,
            "system:time_start": day_start.millis(),
        })

    return ee.ImageCollection(dates.map(build_day))


def to_natural(image):
    """Convert dB backscatter bands to natural (linear power) units."""
    natural = ee.Image(10.0).pow(image.divide(10.0))
    return ee.Image(natural.copyProperties(image, image.propertyNames()))


def to_db(image):
    """Convert natural (linear power) units back to dB."""
    db = ee.Image(image).log10().multiply(10.0)
    return ee.Image(db.copyProperties(image, image.propertyNames()))


def _refined_lee_single_band(band_natural):
    """Refined Lee speckle filter for a single band in natural (linear) units.
    Adapted from the widely-used GEE community implementation (Lemoine/JRC)."""
    weights3 = ee.List.repeat(ee.List.repeat(1, 3), 3)
    kernel3 = ee.Kernel.fixed(3, 3, weights3, 1, 1, False)

    mean3 = band_natural.reduceNeighborhood(ee.Reducer.mean(), kernel3)
    variance3 = band_natural.reduceNeighborhood(ee.Reducer.variance(), kernel3)

    sample_weights = ee.List([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])
    sample_kernel = ee.Kernel.fixed(7, 7, sample_weights, 3, 3, False)

    sample_mean = mean3.neighborhoodToBands(sample_kernel)
    sample_var = variance3.neighborhoodToBands(sample_kernel)

    gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs()
    gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs())
    gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs())
    gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs())

    max_gradient = gradients.reduce(ee.Reducer.max())
    gradmask = gradients.eq(max_gradient)
    gradmask = gradmask.addBands(gradmask)

    directions = sample_mean.select(1).subtract(sample_mean.select(4)) \
        .gt(sample_mean.select(4).subtract(sample_mean.select(7))).multiply(1)
    directions = directions.addBands(
        sample_mean.select(6).subtract(sample_mean.select(4))
        .gt(sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2))
    directions = directions.addBands(
        sample_mean.select(3).subtract(sample_mean.select(4))
        .gt(sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3))
    directions = directions.addBands(
        sample_mean.select(0).subtract(sample_mean.select(4))
        .gt(sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4))
    directions = directions.addBands(directions.select(0).Not().multiply(5))
    directions = directions.addBands(directions.select(1).Not().multiply(6))
    directions = directions.addBands(directions.select(2).Not().multiply(7))
    directions = directions.addBands(directions.select(3).Not().multiply(8))

    directions = directions.updateMask(gradmask)
    directions = directions.reduce(ee.Reducer.sum())

    sample_stats = sample_var.divide(sample_mean.multiply(sample_mean))
    sigma_v = sample_stats.toArray().arraySort().arraySlice(0, 0, 5) \
        .arrayReduce(ee.Reducer.mean(), [0]).arrayGet([0])

    rect_weights = ee.List.repeat(ee.List.repeat(0, 7), 3).cat(ee.List.repeat(ee.List.repeat(1, 7), 4))
    diag_weights = ee.List([
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
    ])
    rect_kernel = ee.Kernel.fixed(7, 7, rect_weights, 3, 3, False)
    diag_kernel = ee.Kernel.fixed(7, 7, diag_weights, 3, 3, False)

    dir_mean = band_natural.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1))
    dir_var = band_natural.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1))
    dir_mean = dir_mean.addBands(
        band_natural.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2)))
    dir_var = dir_var.addBands(
        band_natural.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2)))

    for i in range(1, 4):
        dir_mean = dir_mean.addBands(
            band_natural.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i))
            .updateMask(directions.eq(2 * i + 1)))
        dir_var = dir_var.addBands(
            band_natural.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i))
            .updateMask(directions.eq(2 * i + 1)))
        dir_mean = dir_mean.addBands(
            band_natural.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i))
            .updateMask(directions.eq(2 * i + 2)))
        dir_var = dir_var.addBands(
            band_natural.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i))
            .updateMask(directions.eq(2 * i + 2)))

    dir_mean = dir_mean.reduce(ee.Reducer.sum())
    dir_var = dir_var.reduce(ee.Reducer.sum())

    var_x = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigma_v)).divide(sigma_v.add(1.0))
    b = var_x.divide(dir_var)

    result = dir_mean.add(b.multiply(band_natural.subtract(dir_mean)))
    return result


def refined_lee(image, bands=("VV", "VH")):
    """Apply the Refined Lee filter to the given dB bands of a Sentinel-1 image.
    Returns an image with the same band names, still in dB."""
    natural = ee.Image(10.0).pow(image.select(list(bands)).divide(10.0))
    filtered_bands = [
        _refined_lee_single_band(natural.select([b])).rename(b) for b in bands
    ]
    filtered_natural = ee.Image.cat(filtered_bands)
    filtered_db = filtered_natural.log10().multiply(10.0)
    return ee.Image(filtered_db.copyProperties(image, image.propertyNames()))


def despeckle_collection(daily_mosaics):
    """Apply the Refined Lee filter to every image in a daily-mosaic collection,
    preserving the date/time_start properties."""
    def _apply(img):
        img = ee.Image(img)
        filtered = refined_lee(img)
        return filtered.set({
            "date": img.get("date"),
            "system:time_start": img.get("system:time_start"),
        })
    return ee.ImageCollection(daily_mosaics.map(_apply))


def composite_for_window(despeckled_daily, start, end, method="median"):
    """Temporal composite (median or mean) of despeckled daily mosaics over [start, end)."""
    subset = despeckled_daily.filterDate(start, end)
    if method == "median":
        return subset.median()
    if method == "mean":
        return subset.mean()
    raise ValueError(f"Unknown method: {method}")


def monthly_composite(despeckled_daily, year, month, method="median"):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    return composite_for_window(despeckled_daily, start, end, method)


def event_composite(despeckled_daily, event_key, method="median"):
    """event_key: one of FLOOD_EVENTS ('2019', '2020', '2022')."""
    start, end = FLOOD_EVENTS[event_key]
    return composite_for_window(despeckled_daily, start, end, method)


def baseline_composite(despeckled_daily, event_key, method="median"):
    """Dry-season non-flood composite for the same year as event_key."""
    start, end = DRY_SEASON_BASELINES[event_key]
    return composite_for_window(despeckled_daily, start, end, method)
