"""Artefacts the Trustworthiness rubric (NOTES.md 3.1) demands, produced from our own run.

Three of the four rubric sections need something that actually exists rather than prose:

  1. Data & Model Bias  -> error decomposed by latitude band, climate regime, lead and era.
                           lat/lon are banned as *predictive features* (NOTES 2.5); using
                           them for post-hoc error attribution is the whole point here.
  2. Model Transparency -> SHAP values over the champion, plus a figure. The rubric names
                           figures explicitly and only for this section.
  4. Sustainability     -> read back what CodeCarbon recorded around the real training runs.

Everything is computed on the arena holdout (anchors after 2012-06), scored the way the
leaderboard scores: per lead, then reweighted to the test lead mix.
"""
from __future__ import annotations

import json
import sys
import warnings

# The panel is sparse by construction -- GRACE has 22 missing months and the early years are
# thin -- so nanmean over an empty slice is expected, not a bug. Silence it or the log is
# thousands of identical lines.
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd

for p in ("src", "scripts", "research"):
    sys.path.insert(0, p)

import arena_adapter as A
from tws.metrics import per_lead_mse, test_weighted_rmse
import matplotlib
matplotlib.use("Agg")  # headless backend; it must be chosen before pyplot is imported
import matplotlib.pyplot as plt
import pathlib
import shap
import train as _t
from train import training_rows

OUT = "report/artifacts"
SHAP_ROWS = 40_000


def weighted(df, pred="pred"):
    """Test-weighted RMSE over whatever subset is passed, leads renormalised."""
    pl = per_lead_mse(df, pred_col=pred)
    return test_weighted_rmse(pl)


def holdout_with_predictions(cfg):
    df, y, groups = A.load()
    df["pred"] = A._full_predictions(df, cfg)
    df = df[np.isfinite(df["pred"])].copy()
    cells = A._boot()["cells"]
    df["lat"] = cells["lat"].to_numpy()[df["cell_id"].to_numpy()]
    df["lon"] = cells["lon"].to_numpy()[df["cell_id"].to_numpy()]
    return df


def bias_tables(df):
    """Where the error is unevenly distributed -- the rubric's section 1."""
    tabs = {}

    # Latitude band. GRACE's own error structure is latitude-dependent and the land mask
    # is wildly unbalanced, so this is the first place bias should show.
    edges = [-90, -60, -30, -15, 0, 15, 30, 60, 90]
    df["lat_band"] = pd.cut(df["lat"], edges)
    # Climate regime proxy, built from TRAINING months only so it is not a peek at truth.
    panel, ym_min = A._boot()["panel"], A._boot()["ym_min"]
    hist = panel[:, : A.CUT - ym_min + 1]
    cell_sd = np.nanstd(hist, axis=1)
    df["regime"] = pd.qcut(cell_sd[df["cell_id"].to_numpy()], 4,
                           labels=["calm", "moderate", "active", "volatile"])
    df["era"] = (df["anchor_ym"] // 12).astype(int)
    df["has_anchor"] = np.isfinite(df["anchor_tws"])

    for key in ("lat_band", "regime", "lead", "era", "has_anchor"):
        rows = []
        for k, g in df.groupby(key, observed=True):
            if len(g) < 2000:
                continue
            e = g["pred"].to_numpy() - g["y"].to_numpy()
            rows.append(dict(group=str(k), n=len(g),
                             rmse=float(np.sqrt(np.mean(e ** 2))),
                             bias=float(np.mean(e)),
                             sd_y=float(np.std(g["y"])),
                             skill=1 - float(np.sqrt(np.mean(e ** 2))) / float(np.std(g["y"]))))
        tabs[key] = pd.DataFrame(rows)
    return tabs


def shap_table(cfg):
    """TreeExplainer over a stratified sample of training rows."""
    model, feats = A._fit_model(A.CUT, cfg)
    # Build the rows exactly as _fit_model does. The ERA5 champion needs A._covars(cfg): the
    # booted state holds only the supplied covariates, so the old call had no ERA5 columns.
    _t.WITH_PATH[0] = bool(cfg.get("path_features"))
    _t.FEAT_OPTS[0] = A._opts(cfg, phase="train")
    s = A._boot()
    X, _, _ = training_rows(s["panel"], s["ym_min"], A._covars(cfg), A._Wc(cfg), s["cells"],
                            A.CUT, A.CELL_FRAC, seed=11, return_anchors=True,
                            min_target=cfg.get("train_from"))
    X = A._augment(X, cfg, None, None, None)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(SHAP_ROWS, len(X)), replace=False)
    Xs = X.iloc[idx][feats]
    sv = shap.TreeExplainer(model).shap_values(Xs)
    imp = pd.DataFrame({"feature": feats,
                        "mean_abs_shap": np.abs(sv).mean(axis=0)}
                       ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["share"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    return imp, sv, Xs


def shap_figure(sv, Xs, path):
    shap.summary_plot(sv, Xs, max_display=18, show=False, plot_size=(9, 7))
    plt.title("SHAP — direct multi-lead LightGBM, delta target", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def emissions_summary():
    try:
        e = pd.read_csv("research/emissions.csv")
    except FileNotFoundError:
        return None
    return e[["timestamp", "project_name", "duration", "emissions",
              "energy_consumed", "cpu_model"]]


def main():
    """Write the report artefacts: bias tables from a forward holdout, the SHAP table and figure,
    and the emissions summary.
    """
    pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
    cfg = json.load(open("research/champion.json"))
    print("champion:", json.dumps(cfg))

    df = holdout_with_predictions(cfg)
    print(f"holdout {len(df):,} rows, weighted RMSE {weighted(df):.5f}")

    tabs = bias_tables(df)
    with open(f"{OUT}/bias_tables.md", "w") as fh:
        fh.write(f"# Error decomposition\n\nchampion: `{json.dumps(cfg)}`\n\n"
                 f"holdout {len(df):,} rows, test-weighted RMSE {weighted(df):.5f}\n\n")
        for k, t in tabs.items():
            fh.write(f"## by {k}\n\n{t.to_markdown(index=False, floatfmt='.4f')}\n\n")
            print(f"\n--- by {k} ---")
            print(t.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))

    imp, sv, Xs = shap_table(cfg)
    imp.to_csv(f"{OUT}/shap_importance.csv", index=False)
    print("\n--- SHAP top 15 ---")
    print(imp.head(15).to_string(index=False, float_format=lambda v: f"{v:8.5f}"))
    shap_figure(sv, Xs, f"{OUT}/shap_summary.png")
    print(f"\nwrote {OUT}/shap_summary.png")

    e = emissions_summary()
    if e is not None:
        e.to_csv(f"{OUT}/emissions.csv", index=False)
        print(f"\ntotal tracked emissions {e['emissions'].sum()*1000:.3f} g CO2e "
              f"over {e['duration'].sum():.0f}s")


if __name__ == "__main__":
    main()
