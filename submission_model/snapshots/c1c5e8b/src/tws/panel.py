"""Build dense (cell x month) panels from the competition CSVs.

The competition serves long-format rows. Almost everything we want to do -- lagged
anchors, climatology, neighbourhood means, autoregressive rollout -- is far easier on a
dense matrix indexed by (cell, calendar month). This module builds that once and caches
it to parquet.

Two facts drive the design:

1. ``target(cell, t) == TWS_t(cell, t + 1)`` exactly (verified, zero mismatch over the
   1,977,398 train rows whose t+1 is also present). So targets are just the TWS panel
   shifted by one month, and we never store them separately.
2. 22 calendar months inside the train window have no rows at all (GRACE gaps), but the
   ``target`` column of month t-1 still carries their TWS. Folding targets back into the
   panel recovers those months for free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRAIN_CSV = "data/Train.csv"
TEST_CSV = "data/Test.csv"

COVARIATES = [
    "SPEI_01_t",
    "SPEI_03_t",
    "SPEI_06_t",
    "SPEI_12_t",
    "SOIL_MOISTURE_t",
]


def to_ym(times: pd.Series) -> pd.Series:
    """Calendar month as a dense integer, so that consecutive months differ by 1."""
    t = pd.to_datetime(times)
    return t.dt.year * 12 + t.dt.month


def ym_label(ym: int) -> str:
    return f"{(ym - 1) // 12:04d}-{(ym - 1) % 12 + 1:02d}"


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    for df in (train, test):
        df["ym"] = to_ym(df["time"])
    return train, test


def build_cell_index(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """One row per 1-degree land cell, with a stable integer id.

    lat/lon are a lookup key only -- the organizers ban them as predictive features -- so
    they live here, deliberately separated from anything that reaches a model matrix.
    """
    cells = (
        pd.concat([train[["lat", "lon"]], test[["lat", "lon"]]])
        .drop_duplicates()
        .sort_values(["lat", "lon"], ignore_index=True)
    )
    cells["cell_id"] = np.arange(len(cells), dtype=np.int32)
    return cells


def _key(df: pd.DataFrame, cells: pd.DataFrame) -> pd.Series:
    m = df.merge(cells, on=["lat", "lon"], how="left")
    if m["cell_id"].isna().any():
        raise ValueError("row with a cell absent from the cell index")
    return m["cell_id"].astype(np.int32)


def build_tws_panel(train: pd.DataFrame, test: pd.DataFrame, cells: pd.DataFrame):
    """Dense TWS matrix of shape (n_cells, n_months), NaN where unobserved.

    Sources, in increasing order of precedence (later wins, though they agree exactly
    where they overlap):
      * ``target`` of train month t   -> TWS at month t+1  (recovers the 22 gap months)
      * ``TWS_t`` of train month t    -> TWS at month t
      * ``TWS_t`` of test month t     -> TWS at month t, only for unmasked anchor months
    """
    ym_min = int(min(train["ym"].min(), test["ym"].min()))
    ym_max = int(max(train["ym"].max(), test["ym"].max())) + 1  # +1 for the final target
    n_months = ym_max - ym_min + 1
    n_cells = len(cells)

    panel = np.full((n_cells, n_months), np.nan, dtype=np.float32)

    tr_cell = _key(train, cells).to_numpy()
    tr_col = train["ym"].to_numpy() - ym_min
    panel[tr_cell, tr_col + 1] = train["target"].to_numpy(dtype=np.float32)
    panel[tr_cell, tr_col] = train["TWS_t"].to_numpy(dtype=np.float32)

    te = test[~test["TWS_t_masked"].astype(bool)]
    te_cell = _key(te, cells).to_numpy()
    te_col = te["ym"].to_numpy() - ym_min
    panel[te_cell, te_col] = te["TWS_t"].to_numpy(dtype=np.float32)

    return panel, ym_min


def build_covariate_panels(train: pd.DataFrame, test: pd.DataFrame, cells: pd.DataFrame,
                           ym_min: int, n_months: int) -> dict[str, np.ndarray]:
    """One dense (cell x month) matrix per climate covariate.

    These are never masked -- the organizers confirmed only ``TWS_t`` is ever NaN -- so a
    test row always has its own month's SPEI and soil moisture available.
    """
    out = {}
    frames = []
    for df in (train, test):
        f = df[["lat", "lon", "ym"] + COVARIATES].copy()
        f["cell_id"] = _key(df, cells).to_numpy()
        frames.append(f)
    both = pd.concat(frames, ignore_index=True)
    ci = both["cell_id"].to_numpy()
    mi = both["ym"].to_numpy() - ym_min
    for name in COVARIATES:
        mat = np.full((len(cells), n_months), np.nan, dtype=np.float32)
        mat[ci, mi] = both[name].to_numpy(dtype=np.float32)
        out[name] = mat
    return out
