"""Random Forest flood/non-flood classification (Objective 2)."""

import ee

from sar_processing import get_orbit_collection, get_daily_mosaics, despeckle_collection, FLOOD_EVENTS, DRY_SEASON_BASELINES
from features import build_feature_stack, river_distance
from training_labels import generate_training_labels

# Deliberately excludes elevation and river_dist: those bands are used as *label-
# generation constraints* (see training_labels.py) to keep the pseudo-labels
# high-confidence. Including them as classifier features too would let the model
# just re-derive that same threshold rule instead of learning an actual SAR
# backscatter signature -- a circularity that produced a suspicious 100% resubstitution
# accuracy with ~76% of feature importance sitting on river_dist alone. slope is fine to
# keep since it isn't used anywhere in label generation.
FEATURE_BANDS = [
    "VV", "VH", "VH_minus_VV",
    "VV_contrast", "VV_idm", "VV_entropy",
    "slope",
]


def _cheap_stack(event_img, aoi, dem):
    """Elevation/slope/river-distance/VV/VH only -- everything generate_training_labels()
    needs, deliberately excluding the GLCM texture bands. GLCM is a neighborhood
    convolution and is expensive to evaluate region-wide; the label step doesn't use it,
    so building the full stack (with GLCM) just to pick sample locations was wasting most
    of the compute and pushing calls past GEE's interactive-compute timeout."""
    slope = ee.Terrain.slope(dem.select("dem")).rename("slope")
    return ee.Image.cat([
        event_img.select(["VV", "VH"]),
        dem.select("dem").rename("elevation"),
        slope,
        river_distance(aoi),
    ]).clip(aoi)


def _despeckled_composite(aoi, start, end):
    """Despeckled median composite for a narrow date window only -- pulling a full
    year (as an earlier version of this did) forces Refined Lee over 3-4x more dates
    than the event/baseline windows actually need, which was blowing GEE's interactive
    compute timeout."""
    coll = get_orbit_collection(aoi, start=start, end=end)
    daily = get_daily_mosaics(coll, aoi)
    despeckled = despeckle_collection(daily)
    return despeckled.median()


def build_event_stack_and_labels(aoi, dem, year, points_per_class=1500, scale=20):
    """For one flood-event year: pick sample locations cheaply (no GLCM), then extract
    the full feature stack (with GLCM) only at those point locations via sampleRegions.
    Returns (full_feature_stack, samples FeatureCollection)."""
    e_start, e_end = FLOOD_EVENTS[year]
    b_start, b_end = DRY_SEASON_BASELINES[year]

    event_img = _despeckled_composite(aoi, e_start, e_end)
    base_img = _despeckled_composite(aoi, b_start, b_end)

    cheap = _cheap_stack(event_img, aoi, dem)
    label = generate_training_labels(event_img, base_img, aoi, cheap)

    points = label.stratifiedSample(
        numPoints=points_per_class,
        classBand="label",
        region=aoi.bounds(),
        scale=scale,
        seed=42,
        geometries=True,
        tileScale=4,
    )

    full_stack = build_feature_stack(event_img, aoi, dem)
    samples = full_stack.sampleRegions(collection=points, scale=scale, tileScale=4, geometries=True)
    samples = samples.map(lambda f: f.set("event_year", year))
    return full_stack, samples


def collect_pooled_training_samples(aoi, dem, years=("2019", "2020", "2022"), points_per_class=1500):
    """Pool stratified training samples across multiple flood-event years for a more
    general (less single-event-specific) training set."""
    all_samples = None
    for year in years:
        _, samples = build_event_stack_and_labels(aoi, dem, year, points_per_class)
        all_samples = samples if all_samples is None else all_samples.merge(samples)
    return all_samples


def train_test_split(samples, train_fraction=0.7, seed=42):
    samples = samples.randomColumn("rand", seed=seed)
    train = samples.filter(ee.Filter.lt("rand", train_fraction))
    test = samples.filter(ee.Filter.gte("rand", train_fraction))
    return train, test


def train_rf(train_samples, n_trees=100, feature_bands=None):
    feature_bands = feature_bands or FEATURE_BANDS
    classifier = ee.Classifier.smileRandomForest(numberOfTrees=n_trees, seed=42).train(
        features=train_samples,
        classProperty="label",
        inputProperties=feature_bands,
    )
    return classifier


def evaluate(classifier, test_samples):
    classified = test_samples.classify(classifier)
    error_matrix = classified.errorMatrix("label", "classification")
    return error_matrix
