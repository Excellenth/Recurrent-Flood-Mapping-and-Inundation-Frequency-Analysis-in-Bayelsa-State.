"""Feature stack construction for RF flood classification (Objective 2)."""

import ee

GLCM_WINDOW_RADIUS = 3  # -> 7x7 window
DB_MIN, DB_MAX = -25.0, 0.0  # typical Sentinel-1 IW dB range, used to rescale to byte for GLCM


def _to_glcm_byte(band_db):
    """Rescale a dB band to 0-255 so GLCM texture (which needs discrete gray levels) is meaningful."""
    return band_db.clamp(DB_MIN, DB_MAX).subtract(DB_MIN).divide(DB_MAX - DB_MIN).multiply(255).toByte()


def add_glcm_texture(sar_image, band="VV"):
    """Add contrast, homogeneity (idm), and entropy texture bands derived from the given band."""
    byte_img = _to_glcm_byte(sar_image.select(band)).rename(f"{band}_byte")
    glcm = byte_img.glcmTexture(size=GLCM_WINDOW_RADIUS)
    texture = glcm.select([
        f"{band}_byte_contrast",
        f"{band}_byte_idm",
        f"{band}_byte_ent",
    ]).rename([f"{band}_contrast", f"{band}_idm", f"{band}_entropy"])
    return sar_image.addBands(texture)


def river_distance(aoi, permanent_water_threshold=50, max_distance_m=20000, nominal_scale=30):
    """Distance (m) to nearest permanent-water pixel (proxy for river/channel network),
    from JRC Global Surface Water occurrence.

    Uses fastDistanceTransform rather than a kernel-based .distance(): a Euclidean kernel
    sized for max_distance_m in meters turns into a huge pixel kernel once evaluated at
    finer scales (e.g. 20-30m), and GEE caps kernels at 512px -- fastDistanceTransform has
    no such limit since it's a two-pass distance transform, not a moving kernel.

    Important: the water mask must NOT be selfMask()'d first -- distance calculations need
    the explicit 0-valued (non-water) pixels present to measure *from*. Masking them away
    leaves nothing to compute against and silently falls back to the cap everywhere."""
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    river_mask = jrc.select("occurrence").gte(permanent_water_threshold).unmask(0)
    river_mask = river_mask.reproject(crs="EPSG:4326", scale=nominal_scale)

    max_px = int(max_distance_m / nominal_scale)
    dist_px = river_mask.fastDistanceTransform(max_px, "pixels").sqrt()
    dist_m = dist_px.multiply(nominal_scale)
    return dist_m.clamp(0, max_distance_m).clip(aoi).rename("river_dist")


def build_feature_stack(sar_image, aoi, dem):
    """Combine SAR + derived + ancillary bands into the full feature stack used for
    training and classification. `sar_image` must have VV, VH bands in dB."""
    vv = sar_image.select("VV")
    vh = sar_image.select("VH")
    vh_minus_vv = vh.subtract(vv).rename("VH_minus_VV")

    with_texture = add_glcm_texture(sar_image.select(["VV", "VH"]), band="VV")
    texture_bands = with_texture.select(["VV_contrast", "VV_idm", "VV_entropy"])

    slope = ee.Terrain.slope(dem.select("dem")).rename("slope")
    dist = river_distance(aoi)

    stack = ee.Image.cat([
        vv, vh, vh_minus_vv,
        texture_bands,
        dem.select("dem").rename("elevation"),
        slope,
        dist,
    ])
    return stack.clip(aoi)
