"""GLDAS-2.1 Noah panels — a second, independent land surface model.

WHY THIS EXISTS (NOTES 6.74, 6.76)
----------------------------------
ERA5 is the only land surface model in the feature set, and it is also the only feature family
whose value GROWS toward the test period (drift slope t = +2.26, NOTES 6.72). GLDAS Noah is a
different model — different physics, different parameterisation — on exactly our 1 degree grid.
It is the last untried source of genuinely new information.

LEGALITY
--------
GES DISC's own product page: "GLDAS-2.2 product suites use data assimilation (DA), whereas the
GLDAS-2.0 and GLDAS-2.1 products are 'open-loop' (i.e., no data assimilation)." So 2.1
assimilates nothing, GRACE included. Chat #34403 permits "external non-TWS covariates, provided
every source date is at or before t" and bans GRACE-derived total water storage.

**We expose components only and never their sum.** A summed storage proxy is the one
construction a reader could reasonably call "total water storage", and chat #34674 asking
exactly that is unanswered. The components carry the same information and let the model form
its own combination, so there is no reason to take the risk.

GRID
----
GLDAS is 150 x 360 at 1 degree, lat -59.5..89.5 and lon -179.5..179.5; our cells are lat
-55.5..83.5 on the same centres, so the mapping is exact integer indexing with no
interpolation — unlike ERA5, which needed corner handling. 95.5% of our cells have data;
the rest are ocean-masked in GLDAS and stay NaN.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import xarray as xr

DIR = "data/gldas"

# Storage components and the flux terms that move them. Deliberately NOT summed.
FIELDS = {
    "SoilMoi0_10cm_inst":    "sm0_10",
    "SoilMoi10_40cm_inst":   "sm10_40",
    "SoilMoi40_100cm_inst":  "sm40_100",
    "SoilMoi100_200cm_inst": "sm100_200",
    "SWE_inst":              "swe",
    "CanopInt_inst":         "canopy",
    "RootMoist_inst":        "rootmoist",
    "Qs_acc":                "runoff_surface",
    "Qsb_acc":               "runoff_baseflow",   # groundwater; ERA5 exposes no equivalent
    "Qsm_acc":               "snowmelt",
    "Evap_tavg":             "evap",
    "PotEvap_tavg":          "potevap",
}


def load_gldas_panels(cells: pd.DataFrame, ym_min: int, n_months: int,
                      directory: str = DIR,
                      fields: dict[str, str] | None = None) -> dict[str, np.ndarray]:
    """Dense (n_cells, n_months) float32 panels, one per GLDAS field.

    Shape and dtype match :func:`tws.panel.build_covariate_panels`, so the result merges
    straight into the ``covars`` dict and every existing feature — path statistics,
    neighbourhood means, lags — applies unchanged.
    """
    fields = FIELDS if fields is None else fields
    files = sorted(pathlib.Path(directory).glob("GLDAS_NOAH10_M.A*.nc4"))
    if not files:
        raise FileNotFoundError(f"no GLDAS files in {directory}; run scripts/fetch_gldas.py")

    out = {f"gldas_{v}": np.full((len(cells), n_months), np.nan, np.float32)
           for v in fields.values()}
    ri = cj = None
    seen = 0
    for path in files:
        # GLDAS_NOAH10_M.A<YYYYMM>.021.nc4
        stamp = path.name.split(".")[1][1:]
        col = int(stamp[:4]) * 12 + int(stamp[4:6]) - 1 - ym_min
        if not (0 <= col < n_months):
            continue
        with xr.open_dataset(path) as ds:
            if ri is None:
                lat0, lon0 = float(ds.lat[0]), float(ds.lon[0])
                ri = np.rint(cells["lat"].to_numpy() - lat0).astype(int)
                cj = np.rint(cells["lon"].to_numpy() - lon0).astype(int)
                if ri.min() < 0 or ri.max() >= ds.sizes["lat"] or \
                   cj.min() < 0 or cj.max() >= ds.sizes["lon"]:
                    raise ValueError("cell grid does not fit the GLDAS grid")
            for short, name in fields.items():
                if short not in ds.data_vars:
                    continue
                arr = np.asarray(ds[short].squeeze().values, dtype=np.float32)
                out[f"gldas_{name}"][:, col] = arr[ri, cj]
        seen += 1
    if seen == 0:
        raise ValueError("no GLDAS file fell inside the panel's month range")
    return out
