"""The field CNN, per horizon, with ERA5 forcing -- the two things that have ever paid, joined.

Only two mechanisms have ever moved this leaderboard. ERA5 (§6.44) is the only genuinely new
information admitted to the problem, and the convolutional delta field (§6.91) is the only new
model class, worth +0.00383 the afternoon it was built. They have never been combined: the CNN
reads TWS, SPEI and soil moisture, which describe the *outcome* of the water balance, while
ERA5 supplies its *forcing* -- precipitation, evaporation, runoff, snow.

That is the natural pairing for a delta target. `TWS(t+k) - TWS(t)` is, to first order, the
accumulated `tp + e - ro` over the intervening months, and a convolution can learn how that
input routes across a basin instead of being handed a per-cell scalar.

Design follows §6.91 and §6.92 exactly:

  * **One net per horizon, k = 1..7**, trained to predict `TWS(t+k) - TWS(t)` directly. §6.92
    measured the alternative: rolling one net forward drops its own correlation from 0.75 to
    0.59 because each step is fed its own error.
  * **Covariates at the row's own month.** A test row at lead k sits on `a+k-1`, so it gets
    SPEI, soil moisture and ERA5 there -- observed, never masked (#34403), strictly backward
    looking. The 3-month water-balance accumulation ending at that month is given as its own
    channel, because that is the physical quantity and a 7x7 receptive field cannot integrate.
  * **Sized against §6.68.** ~7k parameters per horizon against ~4,225 effective samples, a
    step count fixed in advance, and no local pricing: the holdout column below is a build
    check and the decision is a w=0.04 board probe.

Covers all 280,961 rows, so one probe and one solve price the whole thing.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, "src"); sys.path.insert(0, "research")

import numpy as np
import pandas as pd
import torch

from tws.panel import (build_cell_index, build_covariate_panels, build_tws_panel,
                       load_raw, ym_label)
from tws.era5 import load_era5_panels
from tws.carbon import track
import fieldcnn as FC

SEEDS = int(os.environ.get("DIR2_SEEDS", "8"))
BASE = os.environ.get("DIR2_BASE", "submission/blend_cnn.csv")
OUT = os.environ.get("TWS_OUT", "submission/fielddirect2.csv")
ARCH = os.environ.get("DIR2_ARCH", "net")           # net | wide
# 6.68 says the holdout cannot decide anything here, so it is not worth 24 months of data.
# HOLD>0 prints a build check and trains on the rest; HOLD=0 trains on everything.
HOLD = int(os.environ.get("DIR2_HOLD", "0"))
ERA = ("era5_precipitation", "era5_evaporation", "era5_runoff",
       "era5_temperature", "era5_snow_water", "era5_water_balance",
       "era5_soilwater_1", "era5_soilwater_2", "era5_soilwater_3",
       "era5_soilwater_4")


def main() -> None:
    """Train one field CNN per forecast lead (direct, not rolled forward) and write predictions.

    Rows with no anchor keep the values of the base blend file.
    """
    train, test = load_raw()
    cells = build_cell_index(train, test)
    panel, ym_min = build_tws_panel(train, test, cells)
    covars = build_covariate_panels(train, test, cells, ym_min, panel.shape[1])
    era = load_era5_panels(cells, ym_min, panel.shape[1])
    have = [f for f in ERA if f in era]
    print(f"ERA5 fields available: {have}")
    print(f"architecture: {ARCH}")
    nlat, nlon, li, lo = FC.to_grid(cells)

    def grid(vec):
        g = np.full((nlat, nlon), np.nan, dtype=np.float32)
        g[li, lo] = vec
        return g

    land = np.zeros((nlat, nlon), dtype=np.float32); land[li, lo] = 1.0
    clim = np.zeros((12, nlat, nlon), dtype=np.float32)
    for m in range(12):
        cols = [c for c in range(panel.shape[1]) if (ym_min + c - 1) % 12 == m
                and ym_min + c <= FC.TRAIN_END]
        clim[m] = np.nan_to_num(grid(np.nanmean(panel[:, cols], axis=1)))

    # Standardise the ERA5 panels once, on train months only, so the net sees O(1) inputs.
    lastcol = FC.TRAIN_END - ym_min
    eraz = {}
    for f in have:
        v = era[f][:, :lastcol + 1]
        mu, sd = float(np.nanmean(v)), float(np.nanstd(v)) or 1.0
        eraz[f] = ((era[f] - mu) / sd).astype(np.float32)
    bal = None
    if "era5_water_balance" in eraz:
        bal = eraz["era5_water_balance"]

    def inputs(t, k):
        """Field channels at month t; forcing channels at the row's own month t+k-1."""
        c, cc = t - ym_min, t - ym_min + k - 1
        cur = panel[:, c]
        cm = (t - 1) % 12
        ch = [grid(cur),
              grid(cur - panel[:, c - 1]),
              grid(cur - panel[:, c - 3]),
              grid(cur - panel[:, c - 12]),
              grid(cur - panel[:, c - 6]),
              grid(cur - panel[:, c - 24]),
              grid(cur) - clim[cm],
              grid(covars["SPEI_03_t"][:, cc]),
              grid(covars["SPEI_12_t"][:, cc]),
              grid(covars["SOIL_MOISTURE_t"][:, cc])]
        ch += [grid(eraz[f][:, cc]) for f in have]
        if bal is not None:
            # TWS is the running integral of the balance, and groundwater responds over
            # years, not a season. A 7x7 kernel cannot integrate, so hand it the integrals.
            for win in (3, 6, 12, 24):
                ch.append(grid(np.nanmean(bal[:, max(0, cc - win + 1):cc + 1], axis=1)))
        ch += [land,
               np.full((nlat, nlon), np.sin(2 * np.pi * (cm + 1) / 12), np.float32),
               np.full((nlat, nlon), np.cos(2 * np.pi * (cm + 1) / 12), np.float32)]
        x = np.stack(ch)
        return np.nan_to_num(x), np.isfinite(x).all(0)

    te = test.merge(cells, on=["lat", "lon"], how="left")
    starts = sorted(te.groupby("ym")["TWS_t_masked"].mean().pipe(lambda s: s[s < 0.5]).index)
    allm = sorted(te["ym"].unique()); run_of = {}
    for i, a in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else max(allm) + 1
        for m in allm:
            if a <= m < end:
                run_of[m] = a
    lead = (te["ym"] - te["ym"].map(run_of) + 1).to_numpy()
    run = te["ym"].map(run_of).to_numpy()
    maxlead = {a: int(lead[run == a].max()) for a in starts}

    base = pd.read_csv(BASE).set_index("ID")["Target"].reindex(te["ID"].to_numpy()).to_numpy()
    out = base.copy()
    cid = te["cell_id"].to_numpy()
    n_done = 0

    with track("tws-fielddirect2"):
        for k in range(1, 8):
            months = [t for t in range(ym_min + 13, FC.TRAIN_END - k)]
            X, Y, M = [], [], []
            for t in months:
                c = t - ym_min
                x, okx = inputs(t, k)
                y = grid(panel[:, c + k] - panel[:, c])
                m = np.isfinite(y) & okx
                if m.sum() < 0.5 * land.sum():
                    continue
                X.append(x); Y.append(np.nan_to_num(y)); M.append(m.astype(np.float32))
            X, Y, M = np.stack(X), np.stack(Y), np.stack(M)
            ntr = len(X) - HOLD if HOLD else len(X)
            Xt = torch.tensor(X[:ntr], device=FC.DEV); Yt = torch.tensor(Y[:ntr], device=FC.DEV)
            Mt = torch.tensor(M[:ntr], device=FC.DEV)
            if HOLD:
                Xh = torch.tensor(X[ntr:], device=FC.DEV)
                Yh = torch.tensor(Y[ntr:], device=FC.DEV)
                Mh = torch.tensor(M[ntr:], device=FC.DEV)

            nets = []
            for s in range(SEEDS):
                torch.manual_seed(1000 * k + s); np.random.seed(1000 * k + s)
                cls = FC.WideNet if ARCH == "wide" else FC.Net
                net = cls(X.shape[1]).to(FC.DEV)
                opt = torch.optim.AdamW(net.parameters(), lr=FC.LR, weight_decay=FC.WD)
                for _ in range(FC.EPOCHS):
                    net.train()
                    loss = ((net(Xt) - Yt) ** 2 * Mt).sum() / Mt.sum()
                    opt.zero_grad(); loss.backward(); opt.step()
                net.eval(); nets.append(net)
            if HOLD:
                with torch.no_grad():
                    bag = torch.stack([n(Xh) for n in nets]).mean(0)
                    hl = (((bag - Yh) ** 2 * Mh).sum() / Mh.sum()).item()
                    pe = ((Yh ** 2 * Mh).sum() / Mh.sum()).item()
                print(f"  k={k}: {len(X)} fields, {X.shape[1]} channels, holdout {hl:.4f} "
                      f"(persistence {pe:.4f})   <- build check only (6.68)")
            else:
                print(f"  k={k}: {len(X)} fields (all of them), {X.shape[1]} channels, "
                      f"{SEEDS} seeds")

            for a in starts:
                if k > maxlead[a]:
                    continue
                x, _ = inputs(a, k)
                with torch.no_grad():
                    t_ = torch.tensor(x[None], device=FC.DEV)
                    d = torch.stack([n(t_) for n in nets]).mean(0).cpu().numpy()[0]
                sel = (lead == k) & (run == a)
                if not sel.any():
                    continue
                out[sel] = panel[cid[sel], a - ym_min] + d[li[cid[sel]], lo[cid[sel]]]
                n_done += int(sel.sum())

    bad = ~np.isfinite(out)
    if bad.any():
        print(f"  {int(bad.sum())} rows have no anchor field; incumbent retained there")
        out[bad] = base[bad]; n_done -= int(bad.sum())
    print(f"\nrows replaced: {n_done:,} ({n_done/len(out):.1%})")
    print(f"corr with base {np.corrcoef(out, base)[0, 1]:.5f}   "
          f"E[p^2] {np.mean(out**2):.6f} (base {np.mean(base**2):.6f})")
    pd.DataFrame({"ID": te["ID"], "Target": out}).to_csv(OUT, index=False,
                                                         float_format="%.5f")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
