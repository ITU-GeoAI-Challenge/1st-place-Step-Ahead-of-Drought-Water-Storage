"""A convolutional model of the global DELTA field -- the last structurally untried mechanism.

NOTES 6.62 stated the structural gap plainly: the test set is six autoregressive rollouts from
a *fully observed global field*, and every model here reads one row plus a hand-specified
500 km neighbourhood mean. At a run-start month the entire planet's storage state is in hand
and the champion uses a scalar summary of it.

6.65 attacked this with EOFs -- global, linear, rank 20 -- and 6.67 measured that member losing
0.0098 on the board. This is the other half of the space: LOCAL and NONLINEAR. A small
convolution learns the spatial filter that turns a neighbourhood into a one-month tendency,
instead of being handed a mean at a radius chosen by sweep.

### Why it is built small, and why it is never scored locally

6.68 is the governing constraint: the error field carries ~38 spatial degrees of freedom and
~4,225 effective samples, and it killed the sequence model (40k steps -> holdout 0.744 against
persistence 0.728). Three consequences are designed in:

  * **Few parameters.** 16 channels, three layers, ~6k weights shared across 15,715 positions.
    A per-cell model of this field would be hopeless; a translation-equivariant one is the only
    kind whose parameter count does not scale with the field.
  * **Few steps.** 6.68 found the 440-step sequence model beat the 40k-step one. Training is
    capped hard and the schedule is fixed in advance, not tuned.
  * **No local pricing.** 6.68's closing instruction is "stop trying to price members locally".
    The holdout number printed here is a SANITY CHECK for a broken build, never a decision. The
    decision comes from a w=0.04 board probe, which returns exact `h` for ~0.00007 (6.83).

### Scope: lead 1 only

6.36 established the useful object is not the mechanism but the mechanism restricted to where
it competes. The complete field exists only at a run start, so this predicts lead-1 rows --
33.3% of the score -- and the incumbent carries the rest.

Compliance: inputs are TWS at months <= t, SPEI and soil moisture at t (never masked in test,
ruling #34403), and a land mask. No coordinate is a feature; convolution is translation
equivariant, so absolute position is not available to the model. Longitude wraps circularly,
which is geometry, not a coordinate value.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tws.panel import (build_cell_index, build_covariate_panels, build_tws_panel,
                       load_raw, ym_label)
from tws.carbon import track

TRAIN_END = 2015 * 12 + 9
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CH = int(os.environ.get("CNN_CH", "16"))
EPOCHS = int(os.environ.get("CNN_EPOCHS", "30"))
LR = float(os.environ.get("CNN_LR", "3e-3"))
WD = float(os.environ.get("CNN_WD", "1e-3"))
SEED = int(os.environ.get("CNN_SEED", "0"))
HOLD = int(os.environ.get("CNN_HOLD", "24"))       # months held out at the end of train
BASE = os.environ.get("CNN_BASE", "submission/blend_d2_fx.csv")
OUT = os.environ.get("TWS_OUT", "submission/fieldcnn.csv")


def to_grid(cells):
    """Dense (lat, lon) index for the 15,715 land cells. Grid is geometry, not a feature."""
    lats = np.arange(-59.5, 89.5 + 1e-6, 1.0)
    lons = np.arange(-179.5, 179.5 + 1e-6, 1.0)
    li = np.searchsorted(lats, cells["lat"].to_numpy().round(1))
    lo = np.searchsorted(lons, cells["lon"].to_numpy().round(1))
    assert np.allclose(lats[li], cells["lat"].to_numpy()), "lat off grid"
    assert np.allclose(lons[lo], cells["lon"].to_numpy()), "lon off grid"
    return len(lats), len(lons), li, lo


class Net(nn.Module):
    """Circular in longitude, zero-padded in latitude. ~6k parameters."""

    def __init__(self, cin, ch=CH):
        super().__init__()
        self.c1 = nn.Conv2d(cin, ch, 5, padding=0)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=0)
        self.c3 = nn.Conv2d(ch, 1, 1)
        self.act = nn.GELU()

    @staticmethod
    def _pad(x, p):
        x = torch.cat([x[..., -p:], x, x[..., :p]], dim=-1)      # wrap longitude
        return nn.functional.pad(x, (0, 0, p, p))               # zero-pad latitude

    def forward(self, x):
        x = self.act(self.c1(self._pad(x, 2)))
        x = self.act(self.c2(self._pad(x, 1)))
        return self.c3(x)[:, 0]


def build_stack(panel, covars, ym_min, nlat, nlon, li, lo, months, with_target=True):
    """(T, C, lat, lon) inputs and (T, lat, lon) delta targets, plus the valid mask."""
    def grid(vec):
        g = np.full((nlat, nlon), np.nan, dtype=np.float32)
        g[li, lo] = vec
        return g

    land = np.zeros((nlat, nlon), dtype=np.float32)
    land[li, lo] = 1.0
    clim = np.zeros((12, nlat, nlon), dtype=np.float32)
    for m in range(12):
        cols = [c for c in range(panel.shape[1]) if (ym_min + c - 1) % 12 == m
                and ym_min + c <= TRAIN_END]
        clim[m] = grid(np.nanmean(panel[:, cols], axis=1))

    X, Y, M = [], [], []
    for t in months:
        c = t - ym_min
        cur = panel[:, c]
        ok = np.isfinite(cur)
        if with_target:
            ok &= np.isfinite(panel[:, c + 1])
        if ok.mean() < 0.5:
            continue
        cm = (t - 1) % 12
        chans = [grid(cur),
                 grid(cur - panel[:, c - 1]),
                 grid(cur - panel[:, c - 3]),
                 grid(cur - panel[:, c - 12]),
                 grid(cur) - clim[cm],
                 grid(covars["SPEI_03_t"][:, c]),
                 grid(covars["SPEI_12_t"][:, c]),
                 grid(covars["SOIL_MOISTURE_t"][:, c]),
                 land,
                 np.full((nlat, nlon), np.sin(2 * np.pi * (cm + 1) / 12), np.float32),
                 np.full((nlat, nlon), np.cos(2 * np.pi * (cm + 1) / 12), np.float32)]
        x = np.stack(chans)
        y = grid(panel[:, c + 1] - cur) if with_target else np.zeros((nlat, nlon), np.float32)
        m = np.isfinite(y) & np.isfinite(x).all(0)
        X.append(np.nan_to_num(x)); Y.append(np.nan_to_num(y)); M.append(m.astype(np.float32))
    return (np.stack(X), np.stack(Y), np.stack(M))


def main() -> None:
    """Train the field CNN on monthly global TWS fields and write its predictions.

    The last HOLD months are held out as a check. Rows the network does not predict keep the
    values of the BASE blend file.
    """
    torch.manual_seed(SEED); np.random.seed(SEED)
    train, test = load_raw()
    cells = build_cell_index(train, test)
    panel, ym_min = build_tws_panel(train, test, cells)
    covars = build_covariate_panels(train, test, cells, ym_min, panel.shape[1])
    nlat, nlon, li, lo = to_grid(cells)
    print(f"grid {nlat}x{nlon}, {len(cells):,} land cells, device {DEV}")

    fit_months = [t for t in range(ym_min + 13, TRAIN_END)]
    X, Y, M = build_stack(panel, covars, ym_min, nlat, nlon, li, lo, fit_months)
    print(f"{len(X)} monthly fields, {X.shape[1]} channels")
    ntr = len(X) - HOLD
    Xt = torch.tensor(X[:ntr], device=DEV); Yt = torch.tensor(Y[:ntr], device=DEV)
    Mt = torch.tensor(M[:ntr], device=DEV)
    Xh = torch.tensor(X[ntr:], device=DEV); Yh = torch.tensor(Y[ntr:], device=DEV)
    Mh = torch.tensor(M[ntr:], device=DEV)

    net = Net(X.shape[1]).to(DEV)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"{n_par:,} parameters against ~4,225 effective samples (NOTES 6.68)")
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)

    with track("tws-fieldcnn"):
        for ep in range(EPOCHS):
            net.train()
            p = net(Xt)
            loss = ((p - Yt) ** 2 * Mt).sum() / Mt.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 10 == 0 or ep == EPOCHS - 1:
                net.eval()
                with torch.no_grad():
                    ph = net(Xh)
                    hl = (((ph - Yh) ** 2 * Mh).sum() / Mh.sum()).item()
                    pe = ((Yh ** 2 * Mh).sum() / Mh.sum()).item()
                print(f"  ep {ep:3d}  train {loss.item():.4f}  holdout {hl:.4f} "
                      f"(persistence {pe:.4f})   <- SANITY ONLY, not a decision (6.68)")

    # Predict the six run-start months of the test set.
    te = test.merge(cells, on=["lat", "lon"], how="left")
    starts = sorted(te.groupby("ym")["TWS_t_masked"].mean().pipe(lambda s: s[s < 0.5]).index)
    net.eval()
    pred = {}
    for t in starts:
        Xs, _, _ = build_stack(panel, covars, ym_min, nlat, nlon, li, lo, [t],
                               with_target=False)
        with torch.no_grad():
            d = net(torch.tensor(Xs, device=DEV)).cpu().numpy()[0]
        pred[t] = d
        print(f"  {ym_label(t)}: delta field sd {d[li, lo].std():.4f}")

    base = pd.read_csv(BASE).set_index("ID")["Target"].reindex(te["ID"].to_numpy()).to_numpy()
    out = base.copy()
    n = 0
    for t in starts:
        sel = (te["ym"].to_numpy() == t) & (~te["TWS_t_masked"].to_numpy().astype(bool))
        idx = te.loc[sel, "cell_id"].to_numpy()
        out[sel] = panel[idx, t - ym_min] + pred[t][li[idx], lo[idx]]
        n += int(sel.sum())
    print(f"\nlead-1 rows replaced: {n:,} ({n/len(out):.1%})")
    print(f"corr with base overall {np.corrcoef(out, base)[0, 1]:.5f}, "
          f"on replaced rows {np.corrcoef(out[out != base], base[out != base])[0, 1]:.5f}")
    pd.DataFrame({"ID": te["ID"], "Target": out}).to_csv(OUT, index=False,
                                                         float_format="%.5f")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
