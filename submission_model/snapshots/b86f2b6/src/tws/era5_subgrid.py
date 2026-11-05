"""Sub-grid statistics of the ERA5 water balance, per competition cell.

Our main ERA5 pull is at exactly 1 degree -- the competition grid -- so each cell carries a
single number and within-cell heterogeneity is invisible. This reads the 0.25 degree pull and
summarises the 16 sub-cells inside each 1 degree cell.

Why the distribution and not just the mean: TWS is the integral of storage over the cell, so
its change depends on how the water sits inside it. A cell that is half saturated and half dry
absorbs a wet month quite differently from a uniformly moist one -- the saturated half runs
off, the dry half soaks. `sd` measures exactly that heterogeneity and `wetfrac` measures how
much of the cell is already near its own ceiling. Neither is recoverable from a 1 degree mean.

Legal on the same terms as the existing ERA5 work: a non-TWS covariate dated <= t, with
coordinates used only to sample a grid and never as a feature (NOTES 2.5).
"""
from __future__ import annotations

import pathlib
import tempfile
import zipfile

import numpy as np
import pandas as pd
import xarray as xr

DIR = "data/era5_025"
FIELDS = {"tp": "precipitation", "e": "evaporation", "ro": "runoff", "swvl1": "soilwater_1"}


def _open_year(path):
    """CDS returns a zip of netCDF members even when asked for unarchived files."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z, tempfile.TemporaryDirectory() as td:
            for name in z.namelist():
                if name.endswith(".nc"):
                    z.extract(name, td)
                    yield xr.open_dataset(pathlib.Path(td) / name).load()
    else:
        yield xr.open_dataset(path).load()


def _cell_members(cells: pd.DataFrame, lats: np.ndarray, lons: np.ndarray):
    """For each competition cell, the flat indices of the 0.25 deg points inside it.

    A cell centred at (L, O) spans [L-0.5, L+0.5] x [O-0.5, O+0.5]. Membership is by centre,
    so a point on the boundary belongs to exactly one cell.
    """
    clat = cells["lat"].to_numpy(); clon = (cells["lon"].to_numpy() % 360.0)
    lon_g = lons % 360.0
    ilat = {round(float(v), 3): i for i, v in enumerate(lats)}
    ilon = {round(float(v), 3): i for i, v in enumerate(lon_g)}
    step = 0.25
    offs = np.array([-0.375, -0.125, 0.125, 0.375])
    out = np.full((len(cells), len(offs) * len(offs)), -1, np.int64)
    for k, (dla, dlo) in enumerate([(a, b) for a in offs for b in offs]):
        for i in range(len(cells)):
            la = round(float(np.round((clat[i] + dla) / step) * step), 3)
            lo = round(float(np.round(((clon[i] + dlo) % 360.0) / step) * step) % 360.0, 3)
            ii, jj = ilat.get(la), ilon.get(lo)
            if ii is not None and jj is not None:
                out[i, k] = ii * len(lon_g) + jj
    return out


def load_subgrid_panels(cells: pd.DataFrame, ym_min: int, n_months: int,
                        directory: str = DIR) -> dict[str, np.ndarray]:
    """Per-field sub-grid mean / sd / range / wet-fraction, shaped like every other panel."""
    d = pathlib.Path(directory)
    files = sorted(d.glob("era5q_*.nc"))
    if not files:
        raise FileNotFoundError(f"no 0.25 deg ERA5 in {directory}")
    out: dict[str, np.ndarray] = {}
    members = None
    for path in files:
        for ds in _open_year(path):
            tname = "valid_time" if "valid_time" in ds.coords else "time"
            times = pd.to_datetime(ds[tname].values)
            col = times.year * 12 + times.month - 1 - ym_min
            keep = (col >= 0) & (col < n_months)
            if not keep.any():
                ds.close(); continue
            lats = ds["latitude"].values; lons = ds["longitude"].values
            if members is None:
                members = _cell_members(cells, lats, lons)
                ok = members >= 0
            for short in ds.data_vars:
                if short not in FIELDS:
                    continue
                base = "era5q_" + FIELDS[short]
                arr = np.asarray(ds[short].squeeze().values, np.float32)
                if arr.ndim == 2:
                    arr = arr[None, ...]
                flat = arr.reshape(arr.shape[0], -1)
                vals = np.where(ok[None, :, :], flat[:, np.clip(members, 0, None)], np.nan)
                with np.errstate(invalid="ignore"):
                    mu = np.nanmean(vals, axis=2)
                    sd = np.nanstd(vals, axis=2)
                    rng = np.nanmax(vals, axis=2) - np.nanmin(vals, axis=2)
                    med = np.nanmedian(vals, axis=2, keepdims=True)
                    wet = np.nanmean((vals > med).astype(np.float32), axis=2)
                for suf, v in (("_sd", sd), ("_range", rng), ("_wetfrac", wet), ("_mean", mu)):
                    name = base + suf
                    if name not in out:
                        out[name] = np.full((len(cells), n_months), np.nan, np.float32)
                    out[name][:, col[keep]] = v[keep].T.astype(np.float32)
            ds.close()
    return out
