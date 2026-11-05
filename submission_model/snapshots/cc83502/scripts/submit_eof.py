"""Production EOF rollout: propagate the field's low-dimensional state, not the row.

The mechanism of NOTES 6.62/6.63/6.65. The test set is six autoregressive rollouts from a
COMPLETE global TWS field (at every run-start month TWS_t is masked for 0.000 of rows), and a
spatial EOF basis fitted on train projects onto an out-of-window field at corr 0.63 -- the basis
transfers, only the coefficients must be re-solved, which is possible precisely because those
six fields are observed.

    1. fit a spatial basis U on train months only
    2. project the observed run-start field onto U      -> PC, measured not extrapolated
    3. propagate the PCs L months with a cross-PC ridge map fitted on train
    4. reconstruct at the cell

Rank 3 with the `var` propagator, which screened best (rho_r +0.0507). Cells absent from the
basis -- not observed in every fit month -- fall back to persistence; that is 6% of rows.

Legal: only observed TWS at months <= the anchor, which ruling #34450 permits explicitly
("spatial / grid / CNN models over the observed TWS field up to month t"). The basis is fitted
on train months alone, so no test-period field informs any other run.
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from tws.panel import build_cell_index, build_tws_panel, load_raw

RANK = int(os.environ.get("EOF_RANK", "3"))
LAM = float(os.environ.get("EOF_LAM", "1.0"))
OUT = os.environ.get("TWS_OUT", "submission/eof.csv")
TRAIN_END = 2015 * 12 + 9

train, test = load_raw()
cells = build_cell_index(train, test)
panel, ym_min = build_tws_panel(train, test, cells)
n_cells, n_months = panel.shape
ymv = ym_min + np.arange(n_months)

obs = np.isfinite(panel).mean(axis=0)
fit_hi = (ymv <= TRAIN_END) & (obs > 0.9)
keep = np.isfinite(panel[:, fit_hi]).all(axis=1)
fit_cols = np.where(fit_hi)[0]
print(f"basis: {fit_hi.sum()} train months x {keep.sum():,} of {n_cells:,} cells, rank {RANK}")

M = panel[keep][:, fit_cols]
mu = M.mean(axis=1, keepdims=True)
U, S, _ = np.linalg.svd(M - mu, full_matrices=False)
U = U[:, :RANK]
print(f"variance explained by {RANK} EOFs: {(S[:RANK]**2).sum()/(S**2).sum():.3f}")

def project(col):
    v = panel[keep, col]
    m = np.isfinite(v)
    if m.sum() < 0.5 * len(v):
        return None
    c, *_ = np.linalg.lstsq(U[m], v[m] - mu[m, 0], rcond=None)
    return c

def harm(mo):
    mo = np.atleast_1d(mo)
    return np.column_stack([np.ones(len(mo)), np.sin(2*np.pi*mo/12), np.cos(2*np.pi*mo/12),
                            np.sin(4*np.pi*mo/12), np.cos(4*np.pi*mo/12)])

PC = np.array([project(c) for c in fit_cols])
pos = {c: i for i, c in enumerate(fit_cols)}
prop = {}
for L in range(1, 8):
    pr = [(pos[c], pos[c + L]) for c in fit_cols if (c + L) in pos]
    src = np.array([a for a, _ in pr]); dst = np.array([b for _, b in pr])
    Ht = harm(ymv[fit_cols[dst]] % 12)
    X = np.column_stack([PC[src], Ht])
    G = X.T @ X + LAM * np.eye(X.shape[1])
    prop[L] = np.column_stack([np.linalg.solve(G, X.T @ PC[dst, j]) for j in range(RANK)])
    print(f"  lead {L}: {len(pr)} training pairs")

# ---- test run decomposition (NOTES 6.62) ----------------------------------------------
anch = sorted(test.groupby("ym")["TWS_t_masked"].mean().pipe(lambda s: s[s < 0.5]).index)
allm = sorted(test["ym"].unique()); run_of = {}
for i, a in enumerate(anch):
    end = anch[i + 1] if i + 1 < len(anch) else max(allm) + 1
    for m in allm:
        if a <= m < end:
            run_of[m] = a
key = test["lat"].round(3).astype(str) + "_" + test["lon"].round(3).astype(str)
ckey = cells["lat"].round(3).astype(str) + "_" + cells["lon"].round(3).astype(str)
cidx = pd.Series(np.arange(n_cells), index=ckey.to_numpy()).reindex(key.to_numpy()).to_numpy()
test = test.assign(_a=test["ym"].map(run_of), _c=cidx)
test["_lead"] = test["ym"] - test["_a"] + 1
print(f"\nruns: {[(int(a), int((test['_a']==a).sum())) for a in anch]}")
print("rows per lead:", test.groupby("_lead").size().to_dict())

kidx = np.full(n_cells, -1); kidx[keep] = np.arange(keep.sum())
c_arr = test["_c"].to_numpy(); a_arr = test["_a"].to_numpy(); l_arr = test["_lead"].to_numpy()
pred = np.full(len(test), np.nan)
for a in anch:
    col = int(a) - ym_min
    pc = project(col)
    if pc is None:
        raise SystemExit(f"run start {a} is not observed enough to project")
    for L in np.unique(l_arr[a_arr == a]):
        h = harm(np.array([(ymv[col] + L) % 12]))[0]
        pcn = prop[int(L)].T @ np.concatenate([pc, h])
        field = mu[:, 0] + U @ pcn
        sel = (a_arr == a) & (l_arr == L)
        ki = kidx[c_arr[sel]]
        v = np.where(ki >= 0, field[np.clip(ki, 0, len(field) - 1)], np.nan)
        pred[sel] = v

fb = ~np.isfinite(pred)
pred[fb] = panel[c_arr[fb], a_arr[fb] - ym_min]          # persistence fallback
print(f"\nfallback rows (cell not in basis): {fb.sum():,} ({fb.mean():.3%})")
still = ~np.isfinite(pred)
if still.any():
    pred[still] = np.nanmedian(pred[np.isfinite(pred)])
    print(f"median fill: {still.sum():,}")

out = pd.DataFrame({"ID": test["ID"].to_numpy(), "Target": pred})
assert len(out) == 280961 and out["Target"].notna().all()
out.to_csv(OUT, index=False)
print(f"wrote {OUT}   mean {pred.mean():+.4f}  sd {pred.std():.4f}")
