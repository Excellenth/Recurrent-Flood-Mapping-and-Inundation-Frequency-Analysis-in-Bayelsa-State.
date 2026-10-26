# Code Summary

What every script in this project does, grouped by the phase it belongs to (see `PROJECT_PLAN.md` for full phase detail). As of 2026-10-26.

## Shared/core modules (`src/`)

Imported by most scripts below rather than run directly.

| File | Purpose |
| --- | --- |
| `sar_processing.py` | Core Sentinel-1 SAR preprocessing: orbit/track selection, daily mosaicking, and the documented flood-event/dry-baseline date windows for each event year. |
| `training_labels.py` | Generates semi-automated RF training labels from the event-vs-baseline SAR anomaly, constrained by elevation and river proximity to keep only high-confidence pseudo-labels. |
| `classification.py` | Defines the RF feature bands and helper functions (cheap feature stack, despeckled composite) used to build and classify SAR flood-detection imagery. |
| `features.py` | Builds the feature stack (GLCM texture bands, river-distance transform) used as RF classifier inputs. |
| `gee_utils.py` | Shared Earth Engine initialization and AOI-loading utilities used across the project's scripts. |
| `lstm_data.py` | Builds lightweight zonal (LGA-level) monthly SAR composites and feature extraction for the LSTM dataset, deliberately skipping per-pixel despeckling/texture/classification. |

## Phase 1 — SAR processing (`notebooks/`)

| File | Purpose |
| --- | --- |
| `export_event_composites.py` | One-off export of despeckled SAR event/baseline composites (VV/VH, dB) for 2019, 2020, and 2022 to Google Drive as GeoTIFFs. |

## Phase 2 — RF classification, frequency map, per-event extent maps (`notebooks/`)

| File | Purpose |
| --- | --- |
| `collect_training_samples.py` | Builds and saves RF training samples per flood-event year (2019/2020/2022) locally as JSON, as a resumable step to avoid losing expensive GEE compute if a later step fails. |
| `export_training_samples.py` | One-off export of RF training samples per flood-event year to GEE table assets via async batch tasks, to avoid interactive-compute timeouts. |
| `train_rf.py` | Trains and evaluates the RF flood classifier (local scikit-learn version plus a GEE-side smileRandomForest) on the saved training samples. |
| `classify_2022_sanity_check.py` | Retrains the RF with the corrected feature set, classifies the full 2022 flood composite, and checks whether the resulting flooded-area estimate is plausible against the known ~1.3M-displaced disaster scale. |
| `build_frequency_map.py` | Classifies every wet-season month (2015–2025) with the trained RF and averages results into a per-pixel chronic-flood-frequency map, with JRC permanent-water and SAR-derived persistent-wetland pixels excluded as separate transparency bands. |
| `frequency_stats.py` | Recomputes the same wet-season flood-frequency image as `build_frequency_map.py` and exports summary statistics to a GEE table asset for direct API readback. |
| `export_flood_extent_maps.py` | Retrains the RF classifier on pooled 2019/2020/2022 samples, classifies the full event-window SAR composite, and exports a per-event flood/no-flood classification GeoTIFF for each documented year. |
| `export_2016_flood_extent.py` | Exports a classified 2016 flood-extent GeoTIFF as a substitute for 2015 in the 5-year-interval series, since 2015's SAR backscatter has an uncorrected calibration offset from the training-era baseline. |
| `export_2024_flood_extent.py` | Exports a per-pixel classified flood extent for 2024 (real SAR data, to fill the LSTM's 2024–2025 test window at pixel level). **Started this session, then its GEE task was cancelled** when scope shifted to the 2026–2030 forecast instead — no `bayelsa_2024_flood_extent.tif` exists yet. |
| `export_5yr_interval_extent_maps.py` | Exports classified flood extent for 2015 and 2025 using the same wet-season window as the documented events, for direct 5-year-interval comparability (2016 substituted for 2015). |
| `preview_flood_extent_maps.py` | Generates a 3-panel side-by-side comparison preview of the classified 2019/2020/2022 flood-extent maps. |
| `preview_5yr_interval_maps.py` | 3-panel comparison figure for the 2016/2020/2025 classified flood-extent maps. |

## Phase 3 — LSTM forecasting (`notebooks/`)

| File | Purpose |
| --- | --- |
| `build_lstm_dataset.py` | Pulls monthly zonal (per-LGA) VV/VH/rainfall time series for 2015–2025 from GEE, one year at a time, and writes them to a CSV for LSTM training. |
| `train_lstm.py` | Trains an LSTM to forecast next-month zonal VH backscatter per LGA from a lookback window of VV/VH/rainfall history, using a chronological train/test split. |
| `lstm_flood_risk_interpretation.py` | Converts the LSTM's raw VH forecast into an interpretable per-zone flood-risk index based on each zone's percentile rank within its own historical VH distribution. |
| `lstm_risk_index_actual_vs_forecast.py` | Compares the LSTM's forecast-based risk index against the same percentile-rank index computed from actual observed VH, to isolate whether a seasonal anomaly is real or a forecasting artifact. |
| `lstm_extrapolate_2030.py` | Extrapolates the LSTM flood-risk index to 2026–2030 as a flat, no-trend "business-as-usual" baseline built purely from historical per-month climatology (not recursive forecasting). |
| `preview_2030_baseline.py` | Plots the 2026–2030 no-trend baseline flood-risk index as a seasonal-cycle figure for the paper. |
| `lstm_lga_choropleth.py` | Joins the LSTM zonal flood-risk index (test-period and 2026–2030 baseline) to LGA polygons to produce a two-panel choropleth map. |
| `lstm_pixel_forecast_2026_2030.py` | **(This session.)** Downscales the LGA-level 2026–2030 LSTM baseline to pixel resolution via an additive delta applied to the wet-season flood-frequency raster — not an independently trained per-pixel model. |
| `preview_pixel_forecast_2026_2030.py` | **(This session.)** 2-panel preview figure comparing observed 2015–2025 flood frequency to the downscaled 2026–2030 pixel forecast. |

## Phase 4 — Validation (`notebooks/`)

| File | Purpose |
| --- | --- |
| `validate_against_s2.py` | Independently validates the SAR flood classification's open-water portion against Sentinel-2 NDWI for a date within the 2022 flood event. |
| `spot_check_disagreement.py` | Pulls Sentinel-2 true-color thumbnails at locations where S2 says dry but SAR says flooded, to check whether these look like flooded forest canopy or genuine misclassification. |
| `lga_zonal_check.py` | Cross-checks whether the SAR classification shows Yenagoa LGA standing out as worst-hit in 2022 (per NEMA/media reports), using a JRC-permanent-water-masked classification. |
| `lga_zonal_check_masked.py` | Reruns the LGA-level 2022 flood ranking locally with both JRC permanent-water and SAR-derived persistent-wetland masks applied, correcting a gap left by `lga_zonal_check.py`. |

## Phase 5 — Geospatial intelligence / embankment siting (`notebooks/`)

| File | Purpose |
| --- | --- |
| `phase5_priority_ranking.py` | Combines flood exposure, LSTM forward-looking risk, and population into an LGA-level priority ranking for embankment/intervention planning. |
| `phase5_embankment_siting.py` | Identifies priority flood-break embankment corridor pixels within the top-3 priority LGAs as locations with both high chronic flood frequency and close river proximity. |
| `phase5_corridor_population_weighting.py` | Ranks embankment-corridor segments within the top-3 priority LGAs by estimated population protected, using buffered WorldPop 2020 raster sums per segment. |
| `preview_embankment_corridors.py` | Regenerates the figure overlaying priority embankment corridors on wet-season flood frequency for the top-3 priority LGAs. |

## Deliverable assembly (`notebooks/`)

| File | Purpose |
| --- | --- |
| `build_geopackage.py` | Consolidates every vector and raster map product (AOI, LGAs, all classified/frequency/forecast rasters) into a single `Bayelsa_Flood_SAR.gpkg` for QGIS/ArcGIS use. |
| `report_comparison_charts.py` | **(This session.)** Generates the two summary charts for `FORECAST_COMPARISON_REPORT.md` — flooded vs. non-flooded % by year, and observed vs. projected LSTM risk index by LGA. |
