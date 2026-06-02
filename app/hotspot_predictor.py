"""
Spatial ML for predicting crash hotspots beyond observed cluster centers.

Trains on grid cells with crash history, then scores the full study area to
surface locations with high predicted risk but few recorded crashes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


def _grid_step() -> float:
    """~1.1 km cells at this latitude."""
    return 0.01


def _snap(value: float, step: float) -> float:
    """Align coordinates to a shared grid (avoids float merge mismatches)."""
    return round(round(value / step) * step, 5)


def _crash_risk(row: pd.Series) -> float:
    return (
        1.0
        + row.get("Number of Fatalities", 0) * 10
        + row.get("Number of People with Suspected Serious Injury", 0) * 5
        + row.get("Number of People Injured", 0) * 2
    )


def build_grid_cells(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    step: float | None = None,
    padding: float = 0.01,
) -> pd.DataFrame:
    """Lat/lon grid covering the study area with small padding."""
    step = step or _grid_step()
    lat_min -= padding
    lat_max += padding
    lon_min -= padding
    lon_max += padding

    lat_start = _snap(lat_min, step)
    lat_end = _snap(lat_max, step)
    lon_start = _snap(lon_min, step)
    lon_end = _snap(lon_max, step)

    n_lat = int(round((lat_end - lat_start) / step)) + 1
    n_lon = int(round((lon_end - lon_start) / step)) + 1

    lats = [_snap(lat_start + i * step, step) for i in range(n_lat)]
    lons = [_snap(lon_start + j * step, step) for j in range(n_lon)]

    grid = pd.DataFrame(
        [(lat, lon) for lat in lats for lon in lons],
        columns=["lat_area", "lon_area"],
    )
    return grid


def aggregate_crashes(df: pd.DataFrame, step: float | None = None) -> pd.DataFrame:
    """Observed crash counts and severity per grid cell."""
    step = step or _grid_step()
    work = df.dropna(subset=["Latitude", "Longitude"]).copy()
    work["lat_area"] = work["Latitude"].map(lambda v: _snap(v, step))
    work["lon_area"] = work["Longitude"].map(lambda v: _snap(v, step))
    work["point_risk"] = work.apply(_crash_risk, axis=1)

    agg = (
        work.groupby(["lat_area", "lon_area"], as_index=False)
        .agg(
            crash_count=("Document Number", "count"),
            severity_total=("Severity Score", "sum"),
            fatalities=("Number of Fatalities", "sum"),
            injured=("Number of People Injured", "sum"),
            point_risk_sum=("point_risk", "sum"),
        )
    )
    agg["observed_risk"] = (
        agg["crash_count"]
        + agg["fatalities"] * 10
        + agg["injured"] * 2
        + agg["severity_total"] * 0.1
    )
    return agg


def _gaussian_kernel_weights(
    grid: pd.DataFrame, crashes: pd.DataFrame, bandwidth: float
) -> np.ndarray:
    """Distance-weighted crash exposure for each grid cell."""
    if crashes.empty:
        return np.zeros(len(grid))

    lat_g = grid["lat_area"].to_numpy()[:, None]
    lon_g = grid["lon_area"].to_numpy()[:, None]
    lat_c = crashes["Latitude"].to_numpy()[None, :]
    lon_c = crashes["Longitude"].to_numpy()[None, :]
    risk_c = crashes["point_risk"].to_numpy()[None, :]

    dist_sq = (lat_g - lat_c) ** 2 + (lon_g - lon_c) ** 2
    weights = np.exp(-dist_sq / (2 * bandwidth**2))
    return (weights * risk_c).sum(axis=1)


def build_feature_matrix(
    grid: pd.DataFrame,
    observed: pd.DataFrame,
    crash_points: pd.DataFrame,
    step: float | None = None,
) -> pd.DataFrame:
    """Spatial features for each grid cell."""
    step = step or _grid_step()
    merged = grid.merge(observed, on=["lat_area", "lon_area"], how="left")
    for col in [
        "crash_count",
        "severity_total",
        "fatalities",
        "injured",
        "point_risk_sum",
        "observed_risk",
    ]:
        merged[col] = merged[col].fillna(0)

    points = crash_points.dropna(subset=["Latitude", "Longitude"]).copy()
    points["point_risk"] = points.apply(_crash_risk, axis=1)

    merged["kde_near"] = _gaussian_kernel_weights(merged, points, bandwidth=step * 1.5)
    merged["kde_mid"] = _gaussian_kernel_weights(merged, points, bandwidth=step * 3.0)
    merged["kde_far"] = _gaussian_kernel_weights(merged, points, bandwidth=step * 5.0)

    lat_center = merged["lat_area"].mean()
    lon_center = merged["lon_area"].mean()
    merged["dist_from_center"] = np.sqrt(
        (merged["lat_area"] - lat_center) ** 2
        + (merged["lon_area"] - lon_center) ** 2
    )

    return merged


FEATURE_COLUMNS = [
    "lat_area",
    "lon_area",
    "kde_near",
    "kde_mid",
    "kde_far",
    "dist_from_center",
]


def train_risk_model(features: pd.DataFrame) -> tuple[GradientBoostingRegressor, StandardScaler]:
    """
    Train on cells that have crashes so the model learns spatial risk patterns.
    Target is observed_risk in those cells.
    """
    train_mask = features["crash_count"] > 0
    if train_mask.sum() < 20:
        raise ValueError("Not enough crash locations to train the prediction model.")

    x_train = features.loc[train_mask, FEATURE_COLUMNS]
    y_train = features.loc[train_mask, "observed_risk"]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)

    model = GradientBoostingRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        random_state=42,
    )
    model.fit(x_scaled, y_train)
    return model, scaler


def score_grid(
    features: pd.DataFrame,
    model: GradientBoostingRegressor,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Predict risk for every cell in the grid."""
    scored = features.copy()
    x_all = scaler.transform(scored[FEATURE_COLUMNS])
    scored["predicted_risk"] = model.predict(x_all)
    scored["predicted_risk"] = scored["predicted_risk"].clip(lower=0)
    return scored


def find_predicted_hotspots(
    scored: pd.DataFrame,
    known_hotspots: pd.DataFrame,
    *,
    top_n: int = 15,
    max_observed_crashes: int = 3,
    min_predicted_risk: float | None = None,
) -> pd.DataFrame:
    """
    Return cells with high predicted risk but few observed crashes — areas the
    model flags as emerging danger, not cells already listed as top historical hotspots.
    """
    step = _grid_step()

    if min_predicted_risk is None:
        min_predicted_risk = scored["predicted_risk"].quantile(0.70)

    known_cells = {
        (_snap(row["lat_area"], step), _snap(row["lon_area"], step))
        for _, row in known_hotspots.head(15).iterrows()
    }

    def is_known_cell(row: pd.Series) -> bool:
        key = (_snap(row["lat_area"], step), _snap(row["lon_area"], step))
        return key in known_cells

    candidates = scored[~scored.apply(is_known_cell, axis=1)].copy()
    candidates = candidates[candidates["crash_count"] <= max_observed_crashes]

    # Stay near real crash corridors — skip empty padding far from data.
    kde_floor = scored["kde_mid"].quantile(0.20)
    if kde_floor > 0:
        candidates = candidates[candidates["kde_mid"] >= kde_floor]

    if min_predicted_risk is not None:
        above_floor = candidates["predicted_risk"] >= min_predicted_risk
        if above_floor.any():
            candidates = candidates[above_floor]

    if candidates.empty:
        candidates = scored[~scored.apply(is_known_cell, axis=1)].copy()
        candidates = candidates[candidates["crash_count"] <= max_observed_crashes]

    if candidates.empty:
        return pd.DataFrame()

    candidates["risk_gap"] = candidates["predicted_risk"] - candidates["observed_risk"]
    candidates = candidates.sort_values(["risk_gap", "predicted_risk"], ascending=False)

    max_pred = candidates["predicted_risk"].max()
    candidates["ai_confidence"] = (
        (candidates["predicted_risk"] / max_pred).clip(0, 1) if max_pred > 0 else 0.0
    )

    out = candidates.head(top_n).copy()
    out["hotspot_type"] = "AI predicted"
    return out


def predict_emerging_hotspots(
    crash_df: pd.DataFrame,
    known_hotspots: pd.DataFrame,
    *,
    top_n: int = 15,
    max_observed_crashes: int = 2,
    sensitivity: float = 0.85,
) -> pd.DataFrame:
    """
    End-to-end: build grid, train model, return predicted hotspot locations.

    sensitivity: 0–1, higher = stricter (fewer, higher-confidence predictions).
    """
    work = crash_df.dropna(subset=["Latitude", "Longitude"])
    if len(work) < 50:
        return pd.DataFrame()

    step = _grid_step()
    lat_min, lat_max = work["Latitude"].min(), work["Latitude"].max()
    lon_min, lon_max = work["Longitude"].min(), work["Longitude"].max()

    grid = build_grid_cells(lat_min, lat_max, lon_min, lon_max, step=step)
    observed = aggregate_crashes(work, step=step)
    features = build_feature_matrix(grid, observed, work, step=step)

    model, scaler = train_risk_model(features)
    scored = score_grid(features, model, scaler)

    sensitivity = float(np.clip(sensitivity, 0.55, 0.95))
    min_risk = scored["predicted_risk"].quantile(sensitivity)

    return find_predicted_hotspots(
        scored,
        known_hotspots,
        top_n=top_n,
        max_observed_crashes=max_observed_crashes,
        min_predicted_risk=min_risk,
    )
