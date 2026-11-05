"""Re-anchor the test rows whose cell re-enters the record mid-run.

The test set is six autoregressive runs from a fully observed global field, and
`submit_v2.main` anchors every row of a run at that run's FIRST month. For 980 rows that
throws away a later observation of the same cell:

  * 729 rows sit on a cell that is absent from its run's anchor month, so `anchor_tws` is
    NaN, the delta-target prediction inherits the NaN, and the row falls back to that
    cell's last observation before the anchor -- a median of NINE months before the target
    month, where panel-wide persistence MSE is 0.855 against 0.272 at lead 1.
  * 251 more have a valid anchor at the run start but a fresher one inside the run.

The mask rule (masked iff the same cell's previous month is also in the test file) means a
cell is unmasked exactly when it was missing the month before, so these are GRACE dropouts
re-entering, and their own `TWS_t` is given in the row. Using it is strictly backward
looking: month t predicting t+1.

This patches an existing submission rather than regenerating every stack member, because
the affected rows are 0.35% of the file and the members all carry the same defect.

    TWS_IN=submission/blend_lt.csv TWS_OUT=submission/blend_lt_fx.csv \
        .venv/bin/python scripts/fix_anchor.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "research")

from tws.features import neighbour_matrix
from tws.panel import build_cell_index, build_covariate_panels, build_tws_panel, load_raw

import arena_adapter as A
import submit_v2 as S

IN = os.environ.get("TWS_IN", "submission/blend_lt.csv")
OUT = os.environ.get("TWS_OUT", "submission/fixed.csv")
SCALE = float(os.environ.get("TWS_SCALE", "1.0"))


def best_anchors(test, panel, ym_min):
    """Latest month in the row's run, at or before the row, where the cell is observed."""
    months = np.array(sorted(test["ym"].unique()))
    runs, cur = [], [months[0]]
    for m in months[1:]:
        if m == cur[-1] + 1:
            cur.append(m)
        else:
            runs.append(cur)
            cur = [m]
    runs.append(cur)

    run_start = {m: r[0] for r in runs for m in r}
    test = test.copy()
    test["run_ym"] = test["ym"].map(run_start)
    ci = test["cell_id"].to_numpy()
    best = np.full(len(test), -1, dtype=np.int64)
    for r in runs:
        a0, a1 = r[0], r[-1]
        sel = np.flatnonzero(test["run_ym"].to_numpy() == a0)
        block = panel[:, a0 - ym_min: a1 - ym_min + 1]
        idx = np.where(np.isfinite(block), np.arange(block.shape[1])[None, :], -1)
        latest = np.maximum.accumulate(idx, axis=1)
        best[sel] = latest[ci[sel], test["ym"].to_numpy()[sel] - a0] + a0
    test["anchor_ym"] = np.where(best > 0, best, test["run_ym"])
    test["lead"] = test["ym"] - test["anchor_ym"] + 1
    return test


def main() -> None:
    """Re-anchor the rows whose cell reappears inside a run, and write the patched file.

    Refits the champion LightGBM, predicts those rows from their latest observed month, scales the
    predictions by TWS_SCALE, replaces them in TWS_IN and writes TWS_OUT with 5 decimals.
    """
    cfg = json.load(open(os.environ.get("TWS_CFG", "research/champion.json")))
    train, test = load_raw()
    cells = build_cell_index(train, test)
    panel, ym_min = build_tws_panel(train, test, cells)
    covars = build_covariate_panels(train, test, cells, ym_min, panel.shape[1])
    A._STATE.update(panel=panel, ym_min=ym_min, covars=covars, cells=cells,
                    W={(500.0, None, 1.0): neighbour_matrix(cells, 500.0)})
    covars = S.covars_for(cfg, covars, cells, ym_min, panel.shape[1])
    state = (panel, ym_min, covars, cells)

    te = test.merge(cells, on=["lat", "lon"], how="left")
    te = best_anchors(te, panel, ym_min).reset_index(drop=True)
    fix = te[te["anchor_ym"] > te["run_ym"]]
    print(f"rows to re-anchor: {len(fix):,} of {len(te):,}")
    print("  new lead distribution:", fix["lead"].value_counts().sort_index().to_dict())
    anc = panel[fix["cell_id"].to_numpy(), fix["anchor_ym"].to_numpy() - ym_min]
    assert np.isfinite(anc).all(), "re-anchor picked an unobserved month"

    model, feats = S.fit(S.TRAIN_END, cfg, state)
    p = S.predict(model, feats, cfg, fix, state, S.TRAIN_END)
    assert np.isfinite(p).all(), "re-anchored prediction is not finite"
    print(f"  re-anchored predictions: mean {p.mean():+.4f} sd {p.std():.4f}")

    sub = pd.read_csv(IN)
    old = sub.set_index("ID").loc[fix["ID"].to_numpy(), "Target"].to_numpy()
    print(f"  replaced values:         mean {old.mean():+.4f} sd {old.std():.4f}")
    print(f"  mean |change|            {np.abs(SCALE * p - old).mean():.4f}")
    sub = sub.set_index("ID")
    sub.loc[fix["ID"].to_numpy(), "Target"] = SCALE * p
    sub = sub.reset_index()
    assert len(sub) == len(test) and sub["ID"].is_unique and np.isfinite(sub["Target"]).all()
    sub.to_csv(OUT, index=False, float_format="%.5f")
    print(f"\nwrote {OUT}  rows={len(sub):,}")


if __name__ == "__main__":
    main()
