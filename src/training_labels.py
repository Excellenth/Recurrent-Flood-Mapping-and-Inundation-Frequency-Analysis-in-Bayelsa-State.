"""Semi-automated training label generation for the RF flood classifier.

No manual digitizing tool is available in this pipeline, so labels are derived from
the event-vs-baseline SAR anomaly (see Phase 1 finding: flood shows as either a VH/VV
*decrease* -- open water, specular -- or an *increase* -- flooded vegetation,
double-bounce), constrained by elevation and river proximity so only physically
plausible, high-confidence pixels are kept. Ambiguous mid-range pixels are dropped
rather than force-labeled (confident-region pseudo-labeling, not exhaustive coverage).

These labels are a starting point for training -- Phase 4 (Sentinel-2 NDWI, Google
Earth, NEMA/SEMA cross-referencing) is where accuracy actually gets validated.
"""

import ee

ELEVATION_THRESH_M = 10       # flood-plausible pixels must be at/below this elevation
NEAR_RIVER_M = 2000           # "near river" cutoff for flood-plausible pixels
FAR_RIVER_M = 3000            # "far from river" cutoff for high-confidence non-flood

# Asymmetric percentiles: the VH anomaly distribution is skewed positive (double-bounce
# from flooded vegetation dominates area-wide, per the Phase 1 finding), so a genuine
# open-water *decrease* signal only shows up in the extreme low tail (~p5), while the
# flooded-vegetation *increase* signal is broader (~p90 is already clearly elevated).
ANOMALY_LOW_PERCENTILE = 5    # below this = high-confidence open-water flood signal
ANOMALY_HIGH_PERCENTILE = 90  # above this = high-confidence flooded-vegetation signal


def anomaly_image(event_img, base_img):
    """VV/VH dB difference: event composite minus dry-season baseline composite."""
    anomaly = event_img.select(["VV", "VH"]).subtract(base_img.select(["VV", "VH"]))
    return anomaly.rename(["VV_anomaly", "VH_anomaly"])


def generate_training_labels(event_img, base_img, aoi, feature_stack):
    """Returns an ee.Image with a single 'label' band: 1 = flood, 0 = non-flood,
    masked everywhere else (ambiguous / not sampled). `event_img`/`base_img` should
    be despeckled composites already restricted to their event/baseline windows --
    keep those windows narrow (not a full year) to keep this affordable to evaluate."""
    anomaly = anomaly_image(event_img, base_img)
    vh_anom = anomaly.select("VH_anomaly")

    percentiles = vh_anom.reduceRegion(
        reducer=ee.Reducer.percentile([ANOMALY_LOW_PERCENTILE, ANOMALY_HIGH_PERCENTILE]),
        geometry=aoi, scale=100, maxPixels=1e9, bestEffort=True,
    )
    p_low = ee.Number(percentiles.get(f"VH_anomaly_p{ANOMALY_LOW_PERCENTILE}"))
    p_high = ee.Number(percentiles.get(f"VH_anomaly_p{ANOMALY_HIGH_PERCENTILE}"))

    elevation = feature_stack.select("elevation")
    river_dist = feature_stack.select("river_dist")

    flood_plausible = elevation.lte(ELEVATION_THRESH_M).And(river_dist.lte(NEAR_RIVER_M))
    flood_openwater = vh_anom.lte(p_low).And(flood_plausible)
    flood_vegetation = vh_anom.gte(p_high).And(flood_plausible)
    flood_mask = flood_openwater.Or(flood_vegetation)

    nonflood_mask = river_dist.gte(FAR_RIVER_M)

    label = ee.Image(0).where(flood_mask, 1).updateMask(flood_mask.Or(nonflood_mask))
    return label.rename("label")


def sample_training_points(label_image, feature_stack, aoi, points_per_class=1500, scale=20, seed=42):
    """Stratified sample of the feature stack at labeled pixels.

    Uses aoi.bounds() as the sampling region rather than the true AOI polygon: the
    real Bayelsa boundary is a 34-part MultiPolygon (islands/creek-separated land),
    and passing that directly as stratifiedSample's region was producing "null
    geometry" export failures, likely from GEE's server-side geometry handling
    struggling with a complex multi-part region. The underlying feature_stack/label
    images are still clipped/masked to the true AOI elsewhere, so sampling against
    the bounding box only widens the search extent -- it doesn't let invalid
    (masked) pixels outside the real boundary become samples."""
    combined = feature_stack.addBands(label_image)
    samples = combined.stratifiedSample(
        numPoints=points_per_class,
        classBand="label",
        region=aoi.bounds(),
        scale=scale,
        seed=seed,
        geometries=True,
        tileScale=4,
    )
    return samples
