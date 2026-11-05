"""Two model-free base members: climatology and nb_persist.

This code first ran as an inline command on 2026-09-05 06:49 UTC, at commit f949ddf. It is
saved here unchanged, except that the output folder can be set with TWS_OUT_DIR and the
printed correlation with era5_v5 is skipped when that file is absent.

  climatology  the cell's own mean TWS for the target calendar month, over training months.
  nb_persist   the 500 km neighbourhood mean of the anchor field.

Both use only observed TWS at months up to the anchor. Run from the repository root at
commit f949ddf:
    PYTHONPATH=src python submission_model/scripts/make_climatology_nb.py
"""
import os

import numpy as np
import pandas as pd
from tws.panel import load_raw, build_cell_index, build_tws_panel
from tws.features import neighbour_matrix, neighbour_mean

OUT_DIR = os.environ.get("TWS_OUT_DIR", "submission")

train, test = load_raw()
cells = build_cell_index(train, test)
panel, ym_min = build_tws_panel(train, test, cells)
TRAIN_END = 2015 * 12 + 9

anch = sorted(test.groupby("ym")["TWS_t_masked"].mean().pipe(lambda s: s[s < 0.5]).index)
allm = sorted(test["ym"].unique()); run_of = {}
for i, x in enumerate(anch):
    end = anch[i + 1] if i + 1 < len(anch) else max(allm) + 1
    for m in allm:
        if x <= m < end: run_of[m] = x
key = test["lat"].round(3).astype(str) + "_" + test["lon"].round(3).astype(str)
ckey = cells["lat"].round(3).astype(str) + "_" + cells["lon"].round(3).astype(str)
cidx = pd.Series(np.arange(len(cells)), index=ckey.to_numpy()).reindex(key.to_numpy()).to_numpy()

# --- climatology: cell x calendar-month mean over TRAIN months only -------------------
cols = np.arange(panel.shape[1]); ym = ym_min + cols
hist = ym <= TRAIN_END
clim = np.full((len(cells), 12), np.nan, np.float32)
for cm in range(12):
    sel = hist & (ym % 12 == cm)
    with np.errstate(invalid="ignore"):
        clim[:, cm] = np.nanmean(panel[:, sel], axis=1)
tgt_ym = test["ym"].to_numpy() + 1                      # target is the row's month + 1
pc = clim[cidx, tgt_ym % 12]
gm = np.nanmean(clim)
pc = np.where(np.isfinite(pc), pc, gm)

# --- neighbourhood persistence: 500 km mean of the anchor field ------------------------
W = neighbour_matrix(cells, 500.0)
acol = test["ym"].map(run_of).to_numpy() - ym_min
pn = np.full(len(test), np.nan)
for c in np.unique(acol):
    nb = neighbour_mean(W, panel[:, c])
    m = acol == c
    pn[m] = nb[cidx[m]]
a = panel[cidx, acol]
pn = np.where(np.isfinite(pn), pn, np.where(np.isfinite(a), a, 0.0))

champ_path = os.path.join(OUT_DIR, "era5_v5.csv")
champ = (pd.read_csv(champ_path).set_index("ID")["Target"].reindex(test["ID"].to_numpy()).to_numpy()
         if os.path.exists(champ_path) else None)
for nm, v in (("climatology", pc), ("nb_persist", pn)):
    extra = f"   corr with champion {np.corrcoef(v, champ)[0,1]:.4f}" if champ is not None else ""
    print(f"{nm:12s} sd {np.std(v):.4f}{extra}")
    pd.DataFrame({"ID": test["ID"], "Target": v}).to_csv(os.path.join(OUT_DIR, f"{nm}.csv"), index=False)
print("wrote both")
