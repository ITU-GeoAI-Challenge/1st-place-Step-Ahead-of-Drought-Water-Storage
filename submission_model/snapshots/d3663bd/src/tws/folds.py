"""Mask-aware, forward-in-time validation folds.

The whole point of this module is to make local validation *look like the test set*.
The starter notebook scores RMSE 0.607 on a naive forward split and 0.900 on the
leaderboard; that 0.29 gap is not model error, it is the validation being wrong. It hands
the model ``TWS_t`` on every row while two thirds of the real test set does not have it.

What the test set actually looks like
-------------------------------------
Measured from ``Test.csv``, not assumed. The 18 test months form 6 runs of consecutive
calendar months::

    2015-09                        len 1
    2016-01 2016-02 2016-03        len 3
    2016-06 ... 2016-09            len 4
    2016-12 ... 2017-06            len 7
    2018-07                        len 1
    2018-11 2018-12                len 2

``TWS_t`` is served for the FIRST month of each run and withheld for the rest -- per-month,
not per-row (mask rate is 0.000 on the six anchor months and 0.996-0.999 on the other
twelve; the ~0.3% residue is cells absent from the preceding month). This is exactly the
organizers' stated rule: a row is masked iff the same cell's previous month is also in the
test file.

So a run is an autoregressive rollout. For a row at the k-th month of a run:

    lead k  =  months since the last observed TWS for that cell
    target  =  TWS at (row month + 1)

and the longest run gives lead 7, which is where "1 to 7 months" comes from.

A fold is therefore just an anchor month
----------------------------------------
Given an anchor month ``a`` with L+1 consecutive observable months from ``a``:

    * TWS is observable at every month <= a, and nowhere after it
    * holdout row at month a+k-1 has lead k, target TWS[a+k], for k = 1..L

which mirrors the real thing exactly: the model gets history up to the anchor and must
roll forward blind. Any model fitted for this fold must train only on months <= a.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .panel import to_ym, ym_label

# Row counts per lead in the real test set, measured from Test.csv. Used to weight the
# per-lead validation RMSEs into a single number comparable to the leaderboard.
TEST_LEAD_WEIGHTS: dict[int, int] = {
    1: 93631,
    2: 62503,
    3: 46827,
    4: 31206,
    5: 15638,
    6: 15572,
    7: 15584,
}
MAX_LEAD = 7


@dataclass(frozen=True)
class Fold:
    """One anchor month and the rollout it supports."""

    anchor: int  # ym of the anchor month; TWS observable at every month <= anchor
    max_lead: int  # longest lead this fold can score

    @property
    def label(self) -> str:
        return ym_label(self.anchor)

    def holdout_months(self) -> list[int]:
        """Calendar months the holdout rows sit on, one per lead."""
        return [self.anchor + k - 1 for k in range(1, self.max_lead + 1)]

    def target_month(self, lead: int) -> int:
        return self.anchor + lead


def observable_months(panel: np.ndarray, ym_min: int, min_coverage: float = 0.95) -> np.ndarray:
    """Months where TWS is present for essentially every cell."""
    cov = np.isfinite(panel).mean(axis=0)
    return np.array([ym_min + i for i, c in enumerate(cov) if c >= min_coverage])


def contiguous_runs(months: np.ndarray) -> list[list[int]]:
    if len(months) == 0:
        return []
    runs, cur = [], [int(months[0])]
    for a, b in zip(months, months[1:]):
        if b == a + 1:
            cur.append(int(b))
        else:
            runs.append(cur)
            cur = [int(b)]
    runs.append(cur)
    return runs


def candidate_folds(panel: np.ndarray, ym_min: int, train_end: int,
                    min_lead: int = 3) -> list[Fold]:
    """Every anchor at or before ``train_end`` that supports at least ``min_lead`` leads.

    ``train_end`` is the last month a fold may use, so passing the end of the training
    window keeps every fold strictly inside train and strictly forward-looking.
    """
    obs = observable_months(panel, ym_min)
    obs = obs[obs <= train_end]
    folds = []
    for run in contiguous_runs(obs):
        for i, a in enumerate(run):
            available = len(run) - i - 1  # months after the anchor, each one a target
            lead = min(MAX_LEAD, available)
            if lead >= min_lead:
                folds.append(Fold(anchor=a, max_lead=lead))
    return folds


def select_folds(folds: list[Fold], n: int, require_max_lead: bool = True,
                 min_gap: int = 3) -> list[Fold]:
    """Pick ``n`` late folds that are spread out rather than piled on top of each other.

    Overlapping anchors score overlapping months, so their errors are correlated and a
    pooled RMSE over them understates variance. Spacing anchors by ``min_gap`` months
    keeps the folds closer to independent. Latest first, since recent regime is the best
    proxy for 2015-2018.
    """
    pool = [f for f in folds if (not require_max_lead) or f.max_lead == MAX_LEAD]
    if not pool:
        pool = sorted(folds, key=lambda f: -f.max_lead)
    chosen: list[Fold] = []
    for f in sorted(pool, key=lambda f: -f.anchor):
        if len(chosen) >= n:
            break
        if all(abs(f.anchor - c.anchor) >= min_gap for c in chosen):
            chosen.append(f)
    return sorted(chosen, key=lambda f: f.anchor)


def era_folds(panel: np.ndarray, ym_min: int, era_start: int, train_end: int,
              min_lead: int = 1) -> list[Fold]:
    """Every usable anchor at or after ``era_start`` -- the fold set to actually use.

    Two mistakes are easy here and both were made before this function existed.

    *Requiring a full lead 1..7 rollout from every fold.* That needs 8 consecutive GRACE
    months, and 2011-2015 is mostly 3-month runs, so "the latest 8 full-lead anchors"
    silently reaches back to 2009-2010. Instead take every anchor and let each contribute
    whichever leads it can reach; :func:`tws.metrics.per_lead_mse` pools per lead, so a
    lead-3 fold still informs leads 1-3.

    *Pooling folds across eras.* Persistence skill relative to an all-zero prediction
    degrades monotonically as GRACE ages -- measured at lead 1 it is 0.554 for 2003-2007,
    0.655 for 2008-2010, 0.783 for 2011-2015, against 0.874 on the 2015-2018 leaderboard.
    Mixing eras therefore produces an optimistic number that moves with the fold set
    rather than with the model. Only the latest era is a fair proxy for the test period.
    """
    obs = observable_months(panel, ym_min)
    obs = obs[obs <= train_end]
    out = []
    for run in contiguous_runs(obs):
        for i, a in enumerate(run):
            if a < era_start:
                continue
            lead = min(MAX_LEAD, len(run) - i - 1)
            if lead >= min_lead:
                out.append(Fold(anchor=a, max_lead=lead))
    return sorted(out, key=lambda f: f.anchor)


def build_holdout(fold: Fold, panel: np.ndarray, ym_min: int) -> pd.DataFrame:
    """Long-format holdout rows for one fold.

    Columns: cell_id, ym, lead, anchor_tws (the last observed TWS -- what the model is
    allowed to anchor on), y (the truth). Cells missing either the anchor or the target
    are dropped, mirroring the handful of test rows whose neighbours are absent.
    """
    n_cells = panel.shape[0]
    anchor_col = fold.anchor - ym_min
    anchor_tws = panel[:, anchor_col]

    frames = []
    for lead in range(1, fold.max_lead + 1):
        tcol = fold.target_month(lead) - ym_min
        y = panel[:, tcol]
        ok = np.isfinite(anchor_tws) & np.isfinite(y)
        frames.append(
            pd.DataFrame(
                {
                    "cell_id": np.arange(n_cells, dtype=np.int32)[ok],
                    "ym": np.int32(fold.anchor + lead - 1),
                    "lead": np.int8(lead),
                    "anchor_ym": np.int32(fold.anchor),
                    "anchor_tws": anchor_tws[ok],
                    "y": y[ok],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def describe_test_structure(test_csv: str = "data/Test.csv") -> pd.DataFrame:
    """Re-derive the run/lead structure from Test.csv so it can be asserted, not trusted."""
    te = pd.read_csv(test_csv, usecols=["time", "TWS_t_masked"])
    te["ym"] = to_ym(te["time"])
    per_month = te.groupby("ym").agg(rows=("TWS_t_masked", "size"),
                                     mask_rate=("TWS_t_masked", "mean"))
    months = sorted(per_month.index)
    runs = contiguous_runs(np.array(months))
    out = []
    for run in runs:
        for k, m in enumerate(run, start=1):
            out.append({"ym": m, "month": ym_label(m), "run_len": len(run), "lead": k,
                        "rows": int(per_month.loc[m, "rows"]),
                        "mask_rate": float(per_month.loc[m, "mask_rate"])})
    return pd.DataFrame(out)


# The six test runs, as (gap in months from the end of training data to the run's anchor,
# run length). Measured from Test.csv against a training window ending 2015-09.
TEST_RUNS: list[tuple[int, int]] = [(0, 1), (4, 3), (9, 4), (15, 7), (34, 1), (38, 2)]


def mirror_folds(panel: np.ndarray, ym_min: int, cut: int,
                 train_end: int) -> list[tuple[Fold, int]]:
    """Fold set that mirrors the test set's *staleness*, not just its mask structure.

    ``era_folds`` fixes the era but leaves one mismatch that turned out to dominate: how
    far past its own training data the model is asked to predict. A model trained to
    2009-12 and scored on folds in 2010-2015 sits a mean of 34 months past its data; the
    same model on the real test set sits 84 months past. It was measured on a much fresher
    problem than it faced, and that is the entire 0.062 optimism observed in submission 2.

    So instead of "every anchor in an era", take one anchor per test run, placed at the
    same gap from ``cut`` as that run sits from the end of training, and give it that run's
    length. Six folds, matching the six real runs in gap and lead structure.

    Returns (fold, target_gap) pairs so the achieved gap can be reported against the
    intended one -- GRACE's own gaps mean the nearest feasible anchor is sometimes a month
    or two off, and that should be visible rather than hidden.
    """
    obs = set(int(m) for m in observable_months(panel, ym_min) if m <= train_end)
    out = []
    for gap, run_len in TEST_RUNS:
        want = cut + gap
        best = None
        for delta in range(0, 13):
            for cand in ({want - delta, want + delta} if delta else {want}):
                if all((cand + k) in obs for k in range(run_len + 1)):
                    best = cand
                    break
            if best is not None:
                break
        if best is not None:
            out.append((Fold(anchor=best, max_lead=min(MAX_LEAD, run_len)), gap))
    return out
