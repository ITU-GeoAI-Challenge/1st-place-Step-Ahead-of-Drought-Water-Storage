"""Scoring that mirrors the leaderboard instead of flattering the model.

Never report a single pooled validation RMSE over rows the fold happens to contain: the
fold's lead mix is an artefact of which contiguous month runs the GRACE record offers,
not of the test set. Score each lead separately, then reweight to the test set's own lead
distribution. That reweighted number is the one comparable to the leaderboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .folds import MAX_LEAD, TEST_LEAD_WEIGHTS


def per_lead_mse(df: pd.DataFrame, pred_col: str = "pred", y_col: str = "y") -> pd.DataFrame:
    """Pooled squared error per lead across every fold present in ``df``."""
    err = (df[pred_col].to_numpy(dtype=np.float64) - df[y_col].to_numpy(dtype=np.float64)) ** 2
    g = pd.DataFrame({"lead": df["lead"].to_numpy(), "se": err})
    out = g.groupby("lead").agg(mse=("se", "mean"), n=("se", "size"))
    out["rmse"] = np.sqrt(out["mse"])
    return out.reset_index()


def test_weighted_rmse(per_lead: pd.DataFrame) -> float:
    """Combine per-lead MSEs using the test set's lead row counts.

    Leads the folds could not reach are dropped and the remaining weights renormalised,
    so a fold set that tops out at lead 4 yields an optimistic number -- which is why
    :func:`coverage` is reported alongside it.
    """
    w = per_lead["lead"].map(TEST_LEAD_WEIGHTS).to_numpy(dtype=np.float64)
    if not np.isfinite(w).all() or w.sum() == 0:
        raise ValueError("per-lead table contains leads absent from the test weights")
    mse = per_lead["mse"].to_numpy(dtype=np.float64)
    return float(np.sqrt(np.sum(w * mse) / w.sum()))


def coverage(per_lead: pd.DataFrame) -> float:
    """Fraction of real test rows whose lead this fold set can actually score."""
    total = sum(TEST_LEAD_WEIGHTS.values())
    got = per_lead["lead"].map(TEST_LEAD_WEIGHTS).sum()
    return float(got) / total


def naive_pooled_rmse(df: pd.DataFrame, pred_col: str = "pred", y_col: str = "y") -> float:
    """The number NOT to trust -- kept only to show how far it drifts from the weighted one."""
    e = df[pred_col].to_numpy(dtype=np.float64) - df[y_col].to_numpy(dtype=np.float64)
    return float(np.sqrt(np.mean(e ** 2)))


# Public-leaderboard RMSE of an all-zero submission (forum thread 34375). Squared, it is
# E[y^2] over the public test rows, so it is a direct read of the test period's target
# variance -- which is 28% higher than any window we can hold out of Train.csv.
LB_ZEROS_RMSE = 1.013783458


def projected_lb(weighted: float, zeros_weighted: float) -> float:
    """Rescale a harness RMSE onto the leaderboard's variance.

    The harness reproduces the test set's *structure* (runs, anchors, lead mix) but cannot
    reproduce its *scale*: GRACE noise grows over the mission, so 2016-2018 targets vary
    more than anything in the training window. Absolute RMSEs are therefore not comparable
    to the leaderboard, but the ratio to an all-zero prediction is -- that ratio is
    sqrt(1 - R^2), and the variance cancels out of it.

    Measured against five points, four of ours own submissions or published probes:

        model                 local    LB      k = LB/local
        all zeros             0.8659   1.0138  1.1708
        persistence           0.7563   0.8864  1.1720
        weak model (4 feat)   0.6881   0.8046  1.1693
        0.62x persistence     0.6851   0.8035  1.1729
        RICH model (25 feat)  0.6329   0.8033  1.2693   <-- breaks the law

    **Do not trust this for a feature-rich model.** Four of the five sit on k = 1.170 with
    a spread of 0.004, but the rich model sits at 1.269, i.e. the harness overstates it by
    0.062. The cause is not the harness's mask or lead structure -- those are exact -- but
    the gap between the end of training and the month being predicted. Skill from the rich
    features (neighbourhood, climatology, SPEI) is regime-dependent and decays with that
    gap:

        rich model trained to 2009-12, gap 69-107 mo to test  ->  0.8033
        rich model trained to 2015-09, gap  0-38 mo to test   ->  0.7165

    The folds put a model a mean 34 months past its training data; the submitted refit
    faces only 15. Hence the harness is pessimistic about the refit and optimistic about
    anything trained far from the test window.

    So: use the raw weighted RMSE and :func:`bootstrap_delta` to RANK candidate models --
    that is what the harness is good at, and it resolves a 1% effect. Use this projection
    only as a rough bound, and pin absolute position by actually submitting.
    """
    if zeros_weighted <= 0:
        raise ValueError("zeros baseline must be positive")
    return float(weighted / zeros_weighted * LB_ZEROS_RMSE)


def report(df: pd.DataFrame, name: str = "", pred_col: str = "pred",
           zeros_weighted: float | None = None) -> dict:
    pl = per_lead_mse(df, pred_col=pred_col)
    res = {
        "name": name,
        "weighted_rmse": test_weighted_rmse(pl),
        "naive_pooled_rmse": naive_pooled_rmse(df, pred_col=pred_col),
        "coverage": coverage(pl),
        "n_rows": int(len(df)),
        "per_lead": {int(r.lead): round(float(r.rmse), 6) for r in pl.itertuples()},
        "per_lead_n": {int(r.lead): int(r.n) for r in pl.itertuples()},
    }
    if zeros_weighted is not None:
        res["projected_lb"] = projected_lb(res["weighted_rmse"], zeros_weighted)
    return res


def format_report(res: dict) -> str:
    lines = [f"{res['name']}"]
    lines.append(f"  test-weighted RMSE : {res['weighted_rmse']:.6f}"
                 f"   (lead coverage {res['coverage']*100:.1f}% of test rows)")
    lines.append(f"  naive pooled RMSE  : {res['naive_pooled_rmse']:.6f}   <- do not report this")
    if "projected_lb" in res:
        lines.append(f"  projected LB       : {res['projected_lb']:.6f}"
                     "   <- compare this to the leaderboard")
    per = res["per_lead"]
    lines.append("  per lead           : " + "  ".join(
        f"L{k}={per[k]:.4f}" for k in sorted(per)))
    return "\n".join(lines)


def bootstrap_delta(df: pd.DataFrame, col_a: str, col_b: str, n_boot: int = 2000,
                    seed: int = 0) -> dict:
    """Paired bootstrap over ANCHORS of the test-weighted RMSE difference (a - b).

    Resampling rows would be wrong twice over: rows inside one fold share an anchor month
    and a weather regime, and the same cell appears at every lead. Anchors are the
    independent unit, so that is what gets resampled. A negative interval that excludes
    zero is the only evidence that should promote a challenger over a champion.
    """
    rng = np.random.default_rng(seed)
    anchors = df["anchor_ym"].to_numpy()
    uniq = np.unique(anchors)
    lead = df["lead"].to_numpy()
    y = df["y"].to_numpy(dtype=np.float64)
    ea = (df[col_a].to_numpy(dtype=np.float64) - y) ** 2
    eb = (df[col_b].to_numpy(dtype=np.float64) - y) ** 2

    idx_by_anchor = {a: np.flatnonzero(anchors == a) for a in uniq}
    w = np.array([TEST_LEAD_WEIGHTS[int(l)] for l in range(1, MAX_LEAD + 1)],
                 dtype=np.float64)

    def weighted(sel_err, sel_lead):
        mse = np.array([sel_err[sel_lead == L].mean() if (sel_lead == L).any() else np.nan
                        for L in range(1, MAX_LEAD + 1)])
        ok = np.isfinite(mse)
        return float(np.sqrt(np.sum(w[ok] * mse[ok]) / w[ok].sum()))

    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_anchor[a] for a in pick])
        deltas.append(weighted(ea[sel], lead[sel]) - weighted(eb[sel], lead[sel]))
    d = np.array(deltas)
    point = weighted(ea, lead) - weighted(eb, lead)
    return {
        "delta": point,
        "ci_low": float(np.percentile(d, 2.5)),
        "ci_high": float(np.percentile(d, 97.5)),
        "p_better": float((d < 0).mean()),
        "n_anchors": int(len(uniq)),
    }
