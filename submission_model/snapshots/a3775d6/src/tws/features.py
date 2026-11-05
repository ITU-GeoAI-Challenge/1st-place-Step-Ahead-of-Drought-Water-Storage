"""Feature construction. Every feature here is legal under the 13 Aug ruling.

The ban is on lat/lon *as predictive features* -- raw coordinates, cell ids, coordinate
encodings, spatial embeddings. Coordinates as a **lookup key** are explicitly permitted:
retrieving a cell's own past TWS, defining neighbourhoods, sampling gridded covariates.

So cell identity re-enters the model the legal way -- through the cell's own history
(climatology, sigma, trend) and through its neighbourhood -- and never as a number the
model can memorise a location from. ``build_features`` returns a matrix that contains no
coordinate, and :func:`assert_no_coordinates` enforces that at runtime.

Everything is computed strictly from months <= the anchor, which is the last month whose
TWS the model is allowed to have seen. Covariates (SPEI, soil moisture) are the exception
and are read at the row's OWN month t, because the organizers confirmed only ``TWS_t`` is
ever masked -- every test row has its own month's climate data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

EARTH_RADIUS_KM = 6371.0

BANNED_SUBSTRINGS = ("lat", "lon", "cell_id", "coord")


def assert_no_coordinates(columns) -> None:
    """Fail loudly rather than submit something that gets disqualified in code review."""
    bad = [c for c in columns
           if any(b in c.lower() for b in BANNED_SUBSTRINGS)]
    if bad:
        raise ValueError(f"coordinate-derived features are banned by the rules: {bad}")


def neighbour_matrix(cells: pd.DataFrame, radius_km: float = 500.0,
                     include_self: bool = False) -> sparse.csr_matrix:
    """Row-normalised averaging operator over cells within ``radius_km``.

    Uses a KD-tree on unit-sphere coordinates, so it is correct near the poles and across
    the antimeridian -- both of which a naive lat/lon box gets wrong, and this grid runs to
    83.5N. Coordinates are used only to decide *which cells are neighbours*; the values
    that flow through the operator are TWS, never position.
    """

    lat = np.radians(cells["lat"].to_numpy(dtype=np.float64))
    lon = np.radians(cells["lon"].to_numpy(dtype=np.float64))
    xyz = np.column_stack([np.cos(lat) * np.cos(lon),
                           np.cos(lat) * np.sin(lon),
                           np.sin(lat)])
    chord = 2.0 * np.sin(min(radius_km / EARTH_RADIUS_KM, np.pi) / 2.0)
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(r=chord, output_type="ndarray")
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    if include_self:
        idx = np.arange(len(cells))
        rows = np.concatenate([rows, idx])
        cols = np.concatenate([cols, idx])
    data = np.ones(len(rows), dtype=np.float32)
    W = sparse.csr_matrix((data, (rows, cols)), shape=(len(cells), len(cells)))
    return W


def neighbour_mean(W: sparse.csr_matrix, values: np.ndarray) -> np.ndarray:
    """NaN-aware neighbourhood mean of a single month column."""
    v = np.asarray(values, dtype=np.float32)
    ok = np.isfinite(v).astype(np.float32)
    filled = np.where(np.isfinite(v), v, 0.0).astype(np.float32)
    tot = W @ filled
    cnt = W @ ok
    with np.errstate(invalid="ignore", divide="ignore"):
        out = tot / cnt
    out[cnt == 0] = np.nan
    return out


def climate_sensitivity(panel: np.ndarray, covars: dict, ym_min: int, upto: int,
                        name: str = "SPEI_03_t") -> np.ndarray:
    """Per-cell slope of monthly TWS change on the climate index, from months <= upto.

    The path features tell the model what the climate did between anchor and target. They
    cannot tell it how much *this* cell's storage moves per unit of that climate -- an
    alluvial aquifer and a bare desert respond to the same SPEI anomaly completely
    differently. This regresses dTWS on the index over the cell's own history, which is
    exactly the per-cell statistic the rules permit (derived from the cell's own past, not
    from its coordinates).
    """
    cols = np.arange(panel.shape[1])
    use = cols[(ym_min + cols) <= upto]
    P = panel[:, use]
    C = covars[name][:, use]
    d = np.diff(P, axis=1)
    c = C[:, 1:]
    m = np.isfinite(d) & np.isfinite(c)
    cx = np.where(m, c, 0.0)
    dy = np.where(m, d, 0.0)
    n = m.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm = cx.sum(axis=1) / np.maximum(n, 1)
        dm = dy.sum(axis=1) / np.maximum(n, 1)
        cc = np.where(m, cx - cm[:, None], 0.0)
        dd = np.where(m, dy - dm[:, None], 0.0)
        sxx = (cc * cc).sum(axis=1)
        slope = np.where(sxx > 0, (cc * dd).sum(axis=1) / sxx, 0.0)
    slope[n < 12] = 0.0
    return slope.astype(np.float32)


class HistoryStats:
    """Per-cell statistics computed from months <= ``upto`` only.

    Recomputed per anchor rather than once globally: using a cell's mean over *all* months
    would leak the future, and the organizers named exactly that (cell mean over observed
    test months) as not permitted.
    """

    def __init__(self, panel: np.ndarray, ym_min: int, upto: int):
        """Per-cell statistics from months up to `upto` only: mean, standard deviation, calendar-month
        climatology and a linear trend over the latest observed months.
        """
        self.panel = panel
        self.ym_min = ym_min
        self.upto = upto
        cols = np.arange(panel.shape[1])
        yms = ym_min + cols
        self.use = cols[yms <= upto]
        sub = panel[:, self.use]
        with np.errstate(invalid="ignore"):
            self.mean = np.nanmean(sub, axis=1)
            self.std = np.nanstd(sub, axis=1)
        self.clim = np.full((panel.shape[0], 12), np.nan, dtype=np.float32)
        for cm in range(12):
            sel = self.use[(yms[self.use] - 1) % 12 == cm]
            if len(sel):
                with np.errstate(invalid="ignore"):
                    self.clim[:, cm] = np.nanmean(sub[:, np.searchsorted(self.use, sel)],
                                                  axis=1)
        # linear trend over the most recent 36 observable months
        recent = self.use[-36:]
        if len(recent) >= 6:
            x = (self.ym_min + recent).astype(np.float64)
            x = x - x.mean()
            y = panel[:, recent].astype(np.float64)
            m = np.isfinite(y)
            xm = np.where(m, x[None, :], 0.0)
            ym_ = np.where(m, y, 0.0)
            n = m.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                sxy = (xm * ym_).sum(axis=1)
                sxx = (xm * xm).sum(axis=1)
                self.trend = np.where(sxx > 0, sxy / sxx, 0.0).astype(np.float32)
            self.trend[n < 6] = 0.0
        else:
            self.trend = np.zeros(panel.shape[0], dtype=np.float32)

    def clim_at(self, cell_idx: np.ndarray, ym: np.ndarray) -> np.ndarray:
        return self.clim[cell_idx, ((ym - 1) % 12).astype(int)]


def path_features(cell_idx: np.ndarray, anchor_ym: int, lead_val: int,
                  covars: dict[str, np.ndarray], ym_min: int,
                  rich: bool = False, W_nb=None) -> dict[str, np.ndarray]:
    """Climate over the months BETWEEN the anchor and the row -- the interval we ignore.

    This is the largest unexploited signal in the data. A row at lead 7 has no TWS since
    its anchor, but SPEI and soil moisture are present for *every* month in between: the
    organizers confirmed only ``TWS_t`` is ever masked. Those months are exactly the water
    balance that moved storage from the anchor value to the target, so summarising the
    path should matter most where the model is weakest -- long leads.

    Everything here is dated at or before the row's own month t, so it is strictly
    backward-looking and legal.
    """
    out: dict[str, np.ndarray] = {}
    a = int(anchor_ym) - ym_min
    # The row sits at month t = anchor + lead - 1 and predicts t+1. Only months at or
    # before t may be used, so the window is a+1 .. a+lead-1. An earlier version ran to
    # a+lead, which is t+1 -- the target month's own climate. That is future information
    # and against the rules; at lead 1 it was the entire feature.
    lo, hi = a + 1, a + int(lead_val) - 1
    for name in ("SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SOIL_MOISTURE_t"):
        mat = covars[name]
        lo_c, hi_c = max(lo, 0), min(hi, mat.shape[1] - 1)
        if hi_c < lo_c:
            # lead 1: no months strictly between anchor and row, so use the anchor month.
            block = mat[cell_idx, max(a, 0)][:, None]
        else:
            block = mat[np.ix_(cell_idx, np.arange(lo_c, hi_c + 1))]
        short = name.replace("_t", "")
        with np.errstate(invalid="ignore"):
            out[f"path_{short}_mean"] = np.nanmean(block, axis=1).astype(np.float32)
            out[f"path_{short}_min"] = np.nanmin(block, axis=1).astype(np.float32)
            out[f"path_{short}_max"] = np.nanmax(block, axis=1).astype(np.float32)
        # change across the interval: where the balance ended relative to where it started
        out[f"path_{short}_drift"] = (mat[cell_idx, min(max(hi, 0), mat.shape[1] - 1)]
                                      - mat[cell_idx, max(a, 0)]).astype(np.float32)
        if rich and block.shape[1] < 2:
            # A lead-1 row has a one-month window, so slope and dry-month count are
            # undefined. Emit them anyway: the training matrix contains longer leads and
            # LightGBM needs the same columns at predict time.
            out[f"path_{short}_slope"] = np.zeros(len(cell_idx), dtype=np.float32)
            out[f"path_{short}_ndry"] = (block[:, 0] < -1.0).astype(np.float32)
        elif rich:
            # Slope across the interval, and how long a deficit persisted. A basin that
            # sat at SPEI -1.2 for six months is in a different state from one that
            # averaged -1.2 by spiking once, and the mean cannot tell them apart.
            x = np.arange(block.shape[1], dtype=np.float64)
            x = x - x.mean()
            m = np.isfinite(block)
            bx = np.where(m, x[None, :], 0.0)
            by = np.where(m, block, 0.0)
            sxx = (bx * bx).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"path_{short}_slope"] = np.where(sxx > 0, (bx * by).sum(axis=1) / sxx,
                                                      0.0).astype(np.float32)
            out[f"path_{short}_ndry"] = (block < -1.0).sum(axis=1).astype(np.float32)
        if W_nb is not None:
            # Regional climate drives regional storage; the cell's own series is noisy.
            # The mean is linear, so averaging the path then smoothing spatially equals
            # smoothing then averaging, at a fraction of the cost.
            full = np.full(mat.shape[0], np.nan, dtype=np.float32)
            lo2, hi2 = max(lo, 0), min(max(hi, 0), mat.shape[1] - 1)
            if hi2 < lo2:
                lo2 = hi2 = max(a, 0)
            if hi2 >= lo2:
                with np.errstate(invalid="ignore"):
                    full = np.nanmean(mat[:, lo2:hi2 + 1], axis=1).astype(np.float32)
            out[f"path_{short}_nb"] = neighbour_mean(W_nb, full)[cell_idx].astype(np.float32)
    return out


def build_features(cell_idx: np.ndarray, lead: np.ndarray, anchor_ym: int,
                   panel: np.ndarray, ym_min: int, covars: dict[str, np.ndarray],
                   stats: HistoryStats, W: sparse.csr_matrix,
                   neigh_cache: dict | None = None,
                   with_path: bool = False, opts: dict | None = None) -> pd.DataFrame:
    """Feature frame for rows that all share one anchor month.

    Sharing an anchor is what makes this cheap: the history statistics, the neighbourhood
    means and the lag deltas are all properties of the anchor column, so they are computed
    once and gathered per row.
    """
    opts = opts or {}
    a_col = int(anchor_ym) - ym_min
    cell_idx = np.asarray(cell_idx)
    lead = np.asarray(lead, dtype=np.int64)   # int8 from the holdout frame overflows
    row_ym = int(anchor_ym) + lead - 1     # the month the row sits on
    tgt_ym = int(anchor_ym) + lead         # the month being predicted

    anchor = panel[:, a_col]
    if neigh_cache is None:
        neigh_cache = {}
    if "nb" not in neigh_cache:
        neigh_cache["nb"] = neighbour_mean(W, anchor)
        anom_all = anchor - stats.clim_at(np.arange(panel.shape[0]),
                                          np.full(panel.shape[0], anchor_ym))
        neigh_cache["nb_anom"] = neighbour_mean(W, anom_all)
        neigh_cache["anom_all"] = anom_all
    nb = neigh_cache["nb"]
    nb_anom = neigh_cache["nb_anom"]
    anom_all = neigh_cache["anom_all"]

    lags = tuple((opts or {}).get("lags", (1, 2, 3, 6, 12)))
    obs_cols = (opts or {}).get("observed_cols")

    def lag(k: int) -> np.ndarray:
        """TWS k months before the anchor -- or the nearest observed month before that.

        Fixed offsets are a trap here. Training months are mostly contiguous, so lag 1, 2
        and 3 are nearly always present while fitting; in the test set they are present
        for exactly ONE of the six runs. 2016-06 has nothing at 1, 2, 3 or 6 months back,
        and 2018-07 has nothing at all -- its previous observation is 19 months earlier.
        So a model trained to lean on delta_1m meets NaN on 94.5% of the rows it is
        scored on.

        With ``observed_cols`` supplied, walk back to the nearest month that actually
        exists and report how far that was, so the model can discount a stale lag instead
        of losing the feature entirely. Masking is per-month, not per-cell, so one global
        set of observed columns is exact.
        """
        c = a_col - k
        if c < 0:
            return np.full(panel.shape[0], np.nan, np.float32)
        if obs_cols is not None:
            cand = obs_cols[obs_cols <= c]
            if len(cand) == 0:
                return np.full(panel.shape[0], np.nan, np.float32)
            c = int(cand[-1])
        return panel[:, c]

    def lag_offset(k: int) -> float:
        c = a_col - k
        if obs_cols is None or c < 0:
            return float(k)
        cand = obs_cols[obs_cols <= c]
        return float(a_col - int(cand[-1])) if len(cand) else np.nan

    f = {}
    f["anchor_tws"] = anchor[cell_idx]
    f["lead"] = lead.astype(np.float32)
    for k in lags:
        f[f"delta_{k}m"] = (anchor - lag(k))[cell_idx]
        if obs_cols is not None:
            f[f"lagoff_{k}m"] = np.full(len(cell_idx), lag_offset(k), dtype=np.float32)
    f["cell_mean"] = stats.mean[cell_idx]
    f["cell_std"] = stats.std[cell_idx]
    f["cell_trend"] = stats.trend[cell_idx]
    f["clim_target"] = stats.clim_at(cell_idx, np.broadcast_to(tgt_ym, cell_idx.shape))
    f["clim_anchor"] = stats.clim_at(cell_idx, np.full(len(cell_idx), anchor_ym))
    f["anom_anchor"] = anom_all[cell_idx]
    f["clim_step"] = f["clim_target"] - f["clim_anchor"]
    f["nb_tws"] = nb[cell_idx]
    f["nb_anom"] = nb_anom[cell_idx]
    f["local_dev"] = f["anchor_tws"] - f["nb_tws"]
    f["anom_over_std"] = np.where(f["cell_std"] > 1e-6,
                                  f["anom_anchor"] / f["cell_std"], 0.0)
    # covariates at the row's OWN month -- never masked in the test set
    r_col = np.asarray(row_ym - ym_min).astype(np.int64)
    r_col = np.broadcast_to(r_col, cell_idx.shape)
    in_range = (r_col >= 0) & (r_col < panel.shape[1])
    safe = np.clip(r_col, 0, panel.shape[1] - 1)
    for name, mat in covars.items():
        vals = mat[cell_idx, safe].astype(np.float32)
        f[name] = np.where(in_range, vals, np.nan)
    ang = 2.0 * np.pi * (((np.asarray(tgt_ym) - 1) % 12) + 1) / 12.0
    f["tgt_month_sin"] = np.broadcast_to(np.sin(ang), cell_idx.shape).astype(np.float32)
    f["tgt_month_cos"] = np.broadcast_to(np.cos(ang), cell_idx.shape).astype(np.float32)

    if with_path:
        f.update(path_features(cell_idx, anchor_ym, int(np.max(lead)), covars, ym_min,
                               rich=bool(opts.get("path_rich")),
                               W_nb=opts.get("W_path_nb")))
    pre = (opts or {}).get("pre_anchor_months")
    if pre:
        # What the climate was doing BEFORE the anchor. The anchor TWS is one noisy
        # snapshot; the months leading into it say whether the basin was filling or
        # draining when that snapshot was taken.
        for name in ("SPEI_03_t", "SPEI_12_t", "SOIL_MOISTURE_t"):
            mat = covars[name]
            lo_p, hi_p = max(a_col - int(pre) + 1, 0), max(a_col, 0)
            blk = mat[np.ix_(cell_idx, np.arange(lo_p, hi_p + 1))]
            with np.errstate(invalid="ignore"):
                f[f"pre_{name.replace('_t','')}_mean"] = np.nanmean(blk, axis=1)
            f[f"pre_{name.replace('_t','')}_drift"] = (mat[cell_idx, hi_p]
                                                       - mat[cell_idx, lo_p])

    sens = opts.get("sensitivity")
    if sens is not None:
        f["clim_sensitivity"] = sens[cell_idx]
        for short in ("SPEI_03", "SPEI_06"):
            k = f"path_{short}_mean"
            if k in f:
                f[f"resp_{short}"] = f[k] * f["clim_sensitivity"]

    for radius, Wr in (opts.get("extra_W") or {}).items():
        nb_r = neighbour_mean(Wr, anchor)
        f[f"nb_tws_{int(radius)}"] = nb_r[cell_idx]
        f[f"local_dev_{int(radius)}"] = f["anchor_tws"] - f[f"nb_tws_{int(radius)}"]

    df = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})
    assert_no_coordinates(df.columns)
    return df
