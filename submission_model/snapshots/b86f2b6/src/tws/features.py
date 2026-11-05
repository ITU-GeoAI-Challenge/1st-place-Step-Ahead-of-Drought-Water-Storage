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
                     include_self: bool = False, k: int | None = None,
                     aniso: float = 1.0) -> sparse.csr_matrix:
    """Row-normalised averaging operator over cells within ``radius_km``.

    Uses a KD-tree on unit-sphere coordinates, so it is correct near the poles and across
    the antimeridian -- both of which a naive lat/lon box gets wrong, and this grid runs to
    83.5N. Coordinates are used only to decide *which cells are neighbours*; the values
    that flow through the operator are TWS, never position.

    Two alternatives to the fixed-radius disc, both still coordinates-as-lookup-key:

    ``k`` -- take the k nearest cells instead of everything inside a radius. The grid is
    regular in lat/lon, so a fixed-radius disc contains far more cells near the poles than
    at the equator: the smoothing bandwidth of the current operator varies by roughly an
    order of magnitude with latitude, which is an artefact of the grid rather than a choice.
    A fixed count equalises it. covid19-global-forecasting-week-5's winner used exactly this
    form -- "nearby features from the closest 5, 10 and 20 locations".

    ``aniso`` -- stretch the polar axis before the tree, so the neighbourhood becomes an
    east-west elongated ellipse rather than a disc. Climate fields are anisotropic: zonal
    correlation length exceeds meridional, because weather moves zonally and climate zones
    band by latitude. facebook-v-predicting-check-ins' winner swept seven x-y weighting
    ratios and calls treating the two axes as equivalent the modality's most common trap.
    """

    lat = np.radians(cells["lat"].to_numpy(dtype=np.float64))
    lon = np.radians(cells["lon"].to_numpy(dtype=np.float64))
    xyz = np.column_stack([np.cos(lat) * np.cos(lon),
                           np.cos(lat) * np.sin(lon),
                           np.sin(lat) * float(aniso)])
    chord = 2.0 * np.sin(min(radius_km / EARTH_RADIUS_KM, np.pi) / 2.0)
    tree = cKDTree(xyz)
    if k is not None:
        # +1 because the first neighbour returned is the cell itself
        _, idx = tree.query(xyz, k=int(k) + 1)
        src = np.repeat(np.arange(len(cells)), idx.shape[1] - 1)
        dst = idx[:, 1:].ravel()
        rows, cols = np.concatenate([src, dst]), np.concatenate([dst, src])
    else:
        pairs = tree.query_pairs(r=chord, output_type="ndarray")
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    if include_self:
        idx = np.arange(len(cells))
        rows = np.concatenate([rows, idx])
        cols = np.concatenate([cols, idx])
    data = np.ones(len(rows), dtype=np.float32)
    W = sparse.csr_matrix((data, (rows, cols)), shape=(len(cells), len(cells)))
    if k is not None:
        # the symmetric union double-counts mutual pairs; the operator is a MEAN, so any
        # entry above 1 would silently weight those neighbours twice.
        W.data[:] = 1.0
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


def denoise_panel(panel: np.ndarray, ym_min: int, upto: int, rank: int = 20):
    """Rank-``r`` reconstruction of the TWS panel, fitted on months <= ``upto`` only.

    Every feature we have is LOCAL -- the cell's own history, and a mean over a disc around
    it. None of them can see that 15,715 cells move together: the seasonal cycle, ENSO, the
    global hydrological response to a hot year. A truncated SVD over the (cells x months)
    matrix separates exactly that shared structure from what is specific to one cell, and the
    Walmart winner states the case for it in one line -- "features that are shared across
    different stores are probably signal, while those that are not are more likely to be
    noise". EOF/PCA decomposition is also the standard tool in the GRACE literature this
    target comes from.

    Distinct from the anchor-denoising avenue closed in NOTES 6.3.2. That shrank the anchor
    toward a fitted linear base, and lost because trees already apply better conditional
    shrinkage than a global constant. This adds information the trees do not have: no
    combination of a cell's own history and a local disc mean recovers its projection onto a
    global mode.

    Returns the reconstruction over ALL months. Only the left/right factors are fitted on
    history, so no test-period value enters a statistic a training row can see; the
    reconstruction at a later month is that month's own observed column projected onto the
    historical spatial basis.

    Missing entries are filled with the cell's historical mean before the decomposition --
    the standard treatment, and the panel is 23.7% missing so ignoring it is not an option.
    """
    n_cols = panel.shape[1]
    hist = np.arange(n_cols) <= (upto - ym_min)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(panel[:, hist], axis=1)
    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float64)
    filled = np.where(np.isfinite(panel), panel, mu[:, None]).astype(np.float64)
    centred = filled - mu[:, None]

    # Only months that were actually measured can be projected. 47 of the 201 columns hold
    # nothing at all (the GRACE record's own gaps, including the 11-month GRACE / GRACE-FO
    # break), and filling them with the cell mean before the decomposition then reading the
    # reconstruction back out returns exactly that mean -- so a difference taken against such
    # a month silently becomes a LEVEL, not a difference, for some anchors and not others.
    # That inconsistency cost round 21 a -0.14: the columns are NaN here so callers must
    # handle them, rather than being handed a plausible-looking wrong number.
    seen = np.isfinite(panel).mean(axis=0) > 0.5

    # Spatial basis from history only. Economy SVD on (cells x observed hist months).
    U, S, _ = np.linalg.svd(centred[:, hist & seen], full_matrices=False)
    r = int(min(rank, U.shape[1]))
    B = U[:, :r]                                  # cells x r, the spatial modes
    # Project every observed month -- including later ones -- onto that fixed basis.
    recon = np.full(panel.shape, np.nan, dtype=np.float32)
    recon[:, seen] = (B @ (B.T @ centred[:, seen]) + mu[:, None]).astype(np.float32)
    return recon


PATH_COVARIATES = ("SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SOIL_MOISTURE_t")


def path_features(cell_idx: np.ndarray, anchor_ym: int, lead_val: int,
                  covars: dict[str, np.ndarray], ym_min: int,
                  rich: bool = False, W_nb=None,
                  names: tuple[str, ...] | None = None,
                  W_nb_for: dict | None = None,
                  local_dev: bool = False) -> dict[str, np.ndarray]:
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
    for name in (names or PATH_COVARIATES):
        mat = covars.get(name)
        if mat is None:
            continue
        lo_c, hi_c = max(lo, 0), min(hi, mat.shape[1] - 1)
        if hi_c < lo_c:
            # lead 1: no months strictly between anchor and row, so use the anchor month.
            block = mat[cell_idx, max(a, 0)][:, None]
        else:
            block = mat[np.ix_(cell_idx, np.arange(lo_c, hi_c + 1))]
        short = name[:-2] if name.endswith("_t") else name
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
        # A per-field radius, if one was supplied. 500 km is inherited from the anchor
        # neighbourhood, where it was chosen for STORAGE. Climate fields correlate over
        # longer distances than storage does, so the right "local" for a precipitation
        # anomaly need not be the right one for TWS.
        W_use = (W_nb_for or {}).get(name, W_nb)
        if W_use is not None:
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
            out[f"path_{short}_nb"] = neighbour_mean(W_use, full)[cell_idx].astype(np.float32)
            if local_dev:
                # For the ANCHOR we already give the model `local_dev` = cell minus its
                # neighbourhood, and it is a top-20 feature. The climate path has the
                # neighbourhood mean but never the deviation, so the model can see "the
                # region was wet" but not "this cell was wetter than its region". Those are
                # different statements and only the second is cell-specific.
                out[f"path_{short}_dev"] = (out[f"path_{short}_mean"]
                                            - out[f"path_{short}_nb"]).astype(np.float32)
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

    # Simulated history gap. The six test runs see 6, 8, 11, 14, 30 and 32 of their previous
    # 36 months missing -- the last two sit right after the GRACE / GRACE-FO instrument gap
    # and have NOTHING within 12 months. Every training anchor sees 94.4% of its history and
    # never fewer than 80.6%, so `delta_1m` is present on essentially every training row and
    # absent on 94.5% of scored rows, and the whole delta family is absent on 55% of them.
    # LightGBM sends NaN to a default side learned from training data; a feature that is
    # never missing in training has an uninformed default.
    #
    # ga-customer-revenue-prediction "replicates the test set's exact 168-day/46-day-gap
    # layout four times over history" and mlb-player-digital-engagement-forecasting
    # "simulates the target gap with random offsets". This does the same: each anchor draws
    # one of the six real profiles, deterministically from its own month so the assignment
    # is stable across refits and across the champion/candidate comparison.
    hist_missing = ()
    profiles = (opts or {}).get("hist_mask_profiles")
    if profiles:
        hist_missing = frozenset(profiles[int(anchor_ym) % len(profiles)])

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
        if k in hist_missing:
            if obs_cols is None:
                return np.full(panel.shape[0], np.nan, np.float32)
            # with nearest_lags on, walk back the way the real pipeline does
            nxt = [j for j in range(k + 1, k + 40) if j not in hist_missing]
            k = nxt[0] if nxt else k
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
                               W_nb=opts.get("W_path_nb"),
                               names=opts.get("path_names"),
                               W_nb_for=opts.get("W_path_nb_for"),
                               local_dev=bool(opts.get("path_local_dev"))))
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

    cum = opts.get("cum_names")
    if cum:
        # The INTEGRAL over the anchor->row window, not its mean. TWS is the running
        # integral of (precipitation - evaporation - runoff), so the accumulated anomaly
        # across exactly the months between the anchor and the row is the closest thing to
        # a direct physical estimate of the change the delta target asks for. The mean is
        # already available as path_*_mean; the sum differs because the window LENGTH grows
        # with lead, and a tree cannot form mean x length by splitting -- it has to be
        # handed the product. This is also why the ERA5 gain grows with lead (NOTES 6.8).
        #
        # Same window as path_features: a+1 .. a+lead-1, all dated at or before the row's
        # own month t, never t+1.
        L = int(np.max(lead))
        lo_u, hi_u = a_col + 1, a_col + L - 1
        W_nb_c = opts.get("W_path_nb")
        for name in cum:
            mat = covars.get(name)
            if mat is None:
                continue
            lo_c, hi_c = max(lo_u, 0), min(hi_u, mat.shape[1] - 1)
            if hi_c < lo_c:
                # lead 1: nothing strictly between anchor and row, so the anchor month
                # alone -- consistent with how path_features degenerates.
                block = mat[:, max(a_col, 0)][:, None]
            else:
                block = mat[:, lo_c:hi_c + 1]
            with np.errstate(invalid="ignore"):
                tot = np.nansum(block, axis=1).astype(np.float32)
            f[f"cum_{name}"] = tot[cell_idx]
            if W_nb_c is not None:
                f[f"cum_{name}_nb"] = neighbour_mean(W_nb_c, tot)[cell_idx]

    idx = opts.get("indices")
    if idx:
        # Global climate state, and the per-cell response that makes it usable.
        #
        # An index is one number per month shared by every cell, so on its own a tree can do
        # nothing with it: it has no cross-cell variation and cannot separate the cells the
        # metric scores. The informative part is the INTERACTION -- ENSO does not move
        # Queensland and the Sahel the same way, or even in the same direction.
        #
        # This is also the only information in the whole feature set that sees the planet.
        # Every other feature is the cell, its neighbourhood, or its own history, so a global
        # mode is precisely what none of them can reach.
        #
        # Dated at the row's own month t, which ruling #34403 permits for non-TWS covariates.
        sens = opts.get("index_sens") or {}
        for name, vec in idx.items():
            t_col = np.clip(a_col + lead - 1, 0, len(vec) - 1)
            v = vec[t_col]
            f[f"idx_{name}"] = v.astype(np.float32)
            for lag in (3, 12):
                f[f"idx_{name}_lag{lag}"] = vec[np.clip(t_col - lag, 0, len(vec) - 1)]
            if name in sens:
                sv = sens[name][cell_idx]
                f[f"resp_{name}"] = (sv * v).astype(np.float32)

    sp = opts.get("svd_panel")
    if sp is not None:
        # The globally-shared component of the field, and what is left over. Both are
        # time-varying filtered versions of TWS -- the same class of object as nb_tws, which
        # is a spatially filtered version -- so neither is a cell embedding or a coordinate.
        rec_a = sp[:, a_col]
        f["svd_anchor"] = rec_a[cell_idx]
        f["svd_resid"] = f["anchor_tws"] - f["svd_anchor"]
        for k in (1, 3, 12):
            c = a_col - k
            # sp[:, c] is NaN for an unmeasured month, so the delta propagates NaN rather
            # than degenerating into a level. LightGBM handles the NaN; a wrong number it
            # cannot handle.
            f[f"svd_delta_{k}m"] = ((rec_a - sp[:, c])[cell_idx] if c >= 0
                                    else np.full(len(cell_idx), np.nan, np.float32))
        W_nb_s = opts.get("W_path_nb")
        if W_nb_s is not None:
            f["svd_anchor_nb"] = neighbour_mean(W_nb_s, rec_a)[cell_idx]

    tw = opts.get("tws_trend_windows")
    if tw:
        # Per-cell OLS slope of TWS over the W months ENDING AT the anchor, and its
        # extrapolation slope*lead. Trees hold delta_1m/3m/12m already, but a slope is a
        # linear combination of every month in the window; a tree can only approximate it
        # by stacking splits on individual lags, which costs depth. Measured directly on
        # the panel, slopes add R2 over (anom, delta_12m, delta_3m) by +0.002 at lead 1
        # growing to +0.038 at lead 7 -- the horizon-growing shape that ERA5 also had.
        #
        # The sign is MEAN REVERSION, not persistence: a cell trending up over the past
        # W months tends to fall over the next k (r about -0.13). So this is not a secular
        # trend feature; it is a better-estimated restoring force than anom_anchor alone.
        #
        # Strictly at or before the anchor column, and NaN-aware, so masked test months
        # simply drop out of the fit rather than corrupting it.
        for Wm in tw:
            lo_t = max(a_col - int(Wm) + 1, 0)
            seg = panel[:, lo_t:a_col + 1]
            tt = np.arange(seg.shape[1], dtype=np.float64)
            m = ~np.isnan(seg)
            n = m.sum(1)
            tm = np.where(m, tt, 0.0)
            ym = np.where(m, np.nan_to_num(seg), 0.0)
            st, sy = tm.sum(1), ym.sum(1)
            stt, sty = (tm * tm).sum(1), (tm * ym).sum(1)
            den = n * stt - st * st
            with np.errstate(invalid="ignore", divide="ignore"):
                b = (n * sty - st * sy) / den
            b = np.where((n < 12) | (den <= 0), np.nan, b).astype(np.float32)
            f[f"tws_slope_{int(Wm)}"] = b[cell_idx]
            f[f"tws_slope_{int(Wm)}_x_lead"] = f[f"tws_slope_{int(Wm)}"] * lead
            W_nb_t = opts.get("W_path_nb")
            if W_nb_t is not None:
                f[f"tws_slope_{int(Wm)}_nb"] = neighbour_mean(
                    W_nb_t, np.nan_to_num(b))[cell_idx].astype(np.float32)

    grad_r = opts.get("nb_gradient_radius")
    if grad_r is not None:
        # nb_tws at 500 km says what the region holds; a SECOND radius says whether this
        # cell sits at the centre of that anomaly or on its edge. The difference between two
        # radii is a spatial gradient, which multi-radius features supplied as separate
        # columns could only express as a difference of two splits.
        Wg = opts["nb_gradient_W"]
        nb_far = neighbour_mean(Wg, anchor)
        f["nb_tws_far"] = nb_far[cell_idx]
        f["nb_gradient"] = f["nb_tws"] - f["nb_tws_far"]
        nb_anom_far = neighbour_mean(Wg, anom_all)
        f["nb_anom_gradient"] = f["nb_anom"] - nb_anom_far[cell_idx]

    if opts.get("response_terms"):
        # The same water input does not move every cell equally: a basin with a large
        # storage range absorbs a wet month differently from a desert. The model has the
        # forcing and it has cell_std, but a tree forms products only by stacking splits,
        # which costs depth it spends elsewhere. Hand it the products directly.
        sd = np.maximum(f["cell_std"], 1e-3)
        for src in opts["response_terms"]:
            if src in f:
                f[f"resp_{src}"] = f[src] * sd
                f[f"norm_{src}"] = f[src] / sd
                # forcing interacted with how far the cell already is from normal: a wet
                # anomaly landing on an already-full column has nowhere to go.
                f[f"state_{src}"] = f[src] * f["anom_over_std"]

    sens = opts.get("sensitivity")
    if sens is not None:
        f["clim_sensitivity"] = sens[cell_idx]
        for short in ("SPEI_03", "SPEI_06"):
            k = f"path_{short}_mean"
            if k in f:
                f[f"resp_{short}"] = f[k] * f["clim_sensitivity"]

    for tag, Wa in (opts.get("extra_aniso_W") or {}).items():
        # facebook-v-predicting-check-ins' winner did not pick one x-y weighting ratio -- he
        # counted neighbours under seven of them and handed all seventy columns to the model,
        # "treat the feature space like a hyperparameter space". Round 20 located a single
        # optimum at 2.0 (+0.00158) with the isotropic control at the same neighbour count
        # NEGATIVE (-0.00038), so the shape is what pays; several shapes at once is the
        # untested half of that recipe.
        nb_a = neighbour_mean(Wa, anchor)
        f[f"nb_tws_a{tag}"] = nb_a[cell_idx]
        f[f"local_dev_a{tag}"] = f["anchor_tws"] - f[f"nb_tws_a{tag}"]
        nb_an = neighbour_mean(Wa, anom_all)
        f[f"nb_anom_a{tag}"] = nb_an[cell_idx]

    for radius, Wr in (opts.get("extra_W") or {}).items():
        nb_r = neighbour_mean(Wr, anchor)
        f[f"nb_tws_{int(radius)}"] = nb_r[cell_idx]
        f[f"local_dev_{int(radius)}"] = f["anchor_tws"] - f[f"nb_tws_{int(radius)}"]

    df = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})
    assert_no_coordinates(df.columns)
    return df
