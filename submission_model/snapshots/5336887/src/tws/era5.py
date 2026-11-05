"""ERA5 reanalysis covariates, sampled onto the competition's cell grid.

Why ERA5 at all: TWS is to first order the running integral of precipitation minus
evapotranspiration minus runoff. The competition supplies SPEI and soil moisture, which
describe the *outcome* of that balance. ERA5 supplies the forcing terms. Every other
candidate tried in this project has been a rearrangement of information already in the
files; this is the only genuinely new information admitted to the problem.

Legality (ruling #34403): external non-TWS covariates are permitted provided every source
date is at or before ``t`` for the row using it. Nothing here enforces that -- these are
plain (cell x month) panels, exactly like the supplied covariates, and the constraint lives
where it already lives: in ``features.build_features``, which never reads past the row's own
month. lat/lon are used to *sample* the ERA5 grid and never become features; the panels this
module returns are keyed by ``cell_id`` alone.

Two facts about the grids that matter and that the setup notes got wrong:

* ERA5 at ``grid=[1,1]`` puts its points on **integer** degrees (lat 90..-90, lon 0..359).
  The competition's cells sit on **half** degrees (lat 83.5..-55.5, lon -179.5..179.5). The
  spacing matches but the offset does not, so the cell centres fall exactly in the middle of
  four ERA5 points. Nearest-neighbour sampling would carry a systematic half-cell offset;
  bilinear interpolation at the exact midpoint reduces to the plain mean of the four
  corners, which is what ``_corner_index`` sets up. No interpolation library, no error term
  beyond the intrinsic smoothing.
* CDS returns a **zip** of two netCDF members even when ``download_format`` is
  ``unarchived``, because accumulated and instantaneous fields are written separately.
"""
from __future__ import annotations

import pathlib
import tempfile
import zipfile

import numpy as np
import pandas as pd
import xarray as xr

DIR = "data/era5"

# short name -> what it is. Signs follow ERA5: `e` is NEGATIVE for evaporation from the
# surface, so net input to the column is tp + e - ro.
FIELDS = {
    "tp": "precipitation",
    "e": "evaporation",
    "ro": "runoff",
    "t2m": "temperature",
    "swvl1": "soilwater_1",
    "swvl2": "soilwater_2",
    "swvl3": "soilwater_3",
    "swvl4": "soilwater_4",
    "sd": "snow_water",          # absent from the first pull; handled if it appears
}


def _corner_index(cells: pd.DataFrame, n_lat: int, n_lon: int):
    """Four ERA5 grid indices bracketing each cell centre, as flat indices.

    ERA5 latitudes descend from +90, longitudes run 0..359 eastward. A competition cell at
    lat L, lon O sits midway between latitude rows ``90-L-0.5`` and ``90-L+0.5`` and between
    longitude columns ``(O mod 360) -+ 0.5``.
    """
    lat = cells["lat"].to_numpy(dtype=np.float64)
    lon = cells["lon"].to_numpy(dtype=np.float64)
    i0 = np.rint(90.0 - lat - 0.5).astype(np.int64)
    i1 = i0 + 1
    j0 = np.rint((lon % 360.0) - 0.5).astype(np.int64) % n_lon
    j1 = (j0 + 1) % n_lon
    for a in (i0, i1):
        if a.min() < 0 or a.max() >= n_lat:
            raise ValueError("cell latitude falls outside the ERA5 grid")
    return [(i0, j0), (i0, j1), (i1, j0), (i1, j1)]


def _open_year(path: pathlib.Path):
    """Yield every xarray dataset inside one year's file, zip or bare netCDF."""

    if not zipfile.is_zipfile(path):
        yield xr.open_dataset(path)
        return
    with zipfile.ZipFile(path) as z, tempfile.TemporaryDirectory() as d:
        for name in z.namelist():
            if name.endswith(".nc"):
                yield xr.open_dataset(z.extract(name, d))


def load_era5_panels(cells: pd.DataFrame, ym_min: int, n_months: int,
                     directory: str = DIR) -> dict[str, np.ndarray]:
    """Dense (n_cells, n_months) float32 panels, one per ERA5 field.

    Shape and dtype match :func:`tws.panel.build_covariate_panels` exactly, so the result
    can be merged straight into the ``covars`` dict and every existing feature -- path
    statistics, neighbourhood means, lags -- applies to it unchanged.
    """
    # Two prefixes: the main pull, plus the era5snow_* backfill for the field CDS silently
    # dropped from the first request. Both are read into the same panels.
    d = pathlib.Path(directory)
    files = sorted(d.glob("era5_*.nc")) + sorted(d.glob("era5snow_*.nc"))
    if not files:
        raise FileNotFoundError(f"no ERA5 files in {directory}; run scripts/fetch_era5.py")

    out: dict[str, np.ndarray] = {}
    corners = None
    for path in files:
        for ds in _open_year(path):
            tname = "valid_time" if "valid_time" in ds.coords else "time"
            times = pd.to_datetime(ds[tname].values)
            col = times.year * 12 + times.month - 1 - ym_min
            keep = (col >= 0) & (col < n_months)
            if not keep.any():
                ds.close()
                continue
            n_lat, n_lon = ds.sizes["latitude"], ds.sizes["longitude"]
            if corners is None:
                corners = _corner_index(cells, n_lat, n_lon)

            for short in ds.data_vars:
                if short not in FIELDS:
                    continue
                name = "era5_" + FIELDS[short]
                if name not in out:
                    out[name] = np.full((len(cells), n_months), np.nan, dtype=np.float32)
                # (time, lat, lon) after dropping the singleton number/expver coords
                arr = np.asarray(ds[short].squeeze().values, dtype=np.float32)
                if arr.ndim == 2:                      # a single month
                    arr = arr[None, ...]
                acc = np.zeros((len(times), len(cells)), dtype=np.float64)
                for i, j in corners:
                    acc += arr[:, i, j]
                acc /= len(corners)                    # bilinear at the exact midpoint
                out[name][:, col[keep]] = acc[keep].T.astype(np.float32)
            ds.close()

    # The physical predictor of a storage change: net water into the column. `e` is
    # negative for evaporation in ERA5's convention, so this is input minus losses.
    if {"era5_precipitation", "era5_evaporation", "era5_runoff"} <= set(out):
        out["era5_water_balance"] = (out["era5_precipitation"]
                                     + out["era5_evaporation"]
                                     - out["era5_runoff"]).astype(np.float32)
    return out


def to_anomalies(panels: dict[str, np.ndarray], ym_min: int, upto: int) -> dict[str, np.ndarray]:
    """Replace each field by its deviation from that cell's own monthly climatology.

    TWS is an anomaly, so the raw fields are on the wrong footing: a desert cell and a
    rainforest cell differ far more in level than either does over time. The climatology is
    computed from months at or before ``upto`` only -- the end of the training window -- so
    no test-period value ever enters a statistic that a training row can see.
    """
    n_cols = next(iter(panels.values())).shape[1]
    month_of = (np.arange(n_cols) + ym_min) % 12
    hist = np.arange(n_cols) <= (upto - ym_min)
    out = {}
    for name, mat in panels.items():
        anom = np.full_like(mat, np.nan)
        for m in range(12):
            sel = (month_of == m)
            base = np.nanmean(np.where(sel & hist, mat, np.nan), axis=1, keepdims=True)
            anom[:, sel] = mat[:, sel] - base
        out[name + "_anom"] = anom.astype(np.float32)
    return out
