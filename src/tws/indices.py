"""Global climate indices, and the per-cell sensitivity that makes them usable.

A global index is one number per month, shared by every cell. On its own a tree can do nothing
with it -- it has no cross-cell variation, so it cannot separate the cells the metric scores.
What makes it informative is the INTERACTION: ENSO does not move Queensland and the Sahel the
same way, or even in the same direction.

So each index enters twice: as its own value and lags (global state), and multiplied by a
per-cell sensitivity regressed from that cell's own past dTWS on the index (local response).
The second is the part that carries information the model cannot already reach, because no
feature we ship sees the planet -- every one is the cell, its neighbourhood, or its own history.

Legality. Ruling #34403 permits non-TWS covariates whose every source date is <= t, and these
are observations at month t, not forecasts. Thread 34540's open question was about seasonal
FORECASTS issued at t covering t+1..t+6, which would be future information; an index value at
t is not. The sensitivity is regressed from the cell's own past TWS, the same class of
statistic as cell_trend and climate_sensitivity, both already permitted.
"""
from __future__ import annotations

import numpy as np

FILES = {"nino34": "data/nina34.anom.data", "nao": "data/nao.data",
         "soi": "data/soi.data", "pdo": "data/pdo.data"}


def load_index(path: str) -> dict[int, float]:
    """PSL fixed-width: a header line of years, then 'YYYY v1 .. v12', then trailing notes."""
    out = {}
    for line in open(path):
        parts = line.split()
        if len(parts) != 13:
            continue
        try:
            yr = int(parts[0])
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if not (1800 < yr < 2100):
            continue
        for m, v in enumerate(vals, start=1):
            if v > -99:
                out[yr * 12 + m - 1] = v
    return out


def index_panels(ym_min: int, n_months: int, names=None) -> dict[str, np.ndarray]:
    """One row-vector per index, broadcast over cells by the caller."""
    names = names or list(FILES)
    ym = ym_min + np.arange(n_months)
    out = {}
    for n in names:
        d = load_index(FILES[n])
        v = np.array([d.get(int(m), np.nan) for m in ym], np.float32)
        # forward-fill only from the past, never across a gap from the future
        last = np.nan
        for i in range(len(v)):
            if np.isfinite(v[i]):
                last = v[i]
            else:
                v[i] = last
        out[n] = v
    return out


def cell_sensitivity(panel: np.ndarray, idx: np.ndarray, upto_col: int) -> np.ndarray:
    """Per-cell slope of monthly dTWS on the index, from months <= upto_col only."""
    P = panel[:, :upto_col + 1]
    x = idx[:upto_col + 1]
    d = np.diff(P, axis=1)
    c = np.broadcast_to(x[1:], d.shape)
    m = np.isfinite(d) & np.isfinite(c)
    cx = np.where(m, c, 0.0); dy = np.where(m, d, 0.0)
    n = m.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm = cx.sum(1) / np.maximum(n, 1); dm = dy.sum(1) / np.maximum(n, 1)
        cc = np.where(m, cx - cm[:, None], 0.0); dd = np.where(m, dy - dm[:, None], 0.0)
        sxx = (cc * cc).sum(1)
        slope = np.where(sxx > 0, (cc * dd).sum(1) / sxx, 0.0)
    slope[n < 24] = 0.0
    return slope.astype(np.float32)
