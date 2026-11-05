"""Draw submission_model/architecture.png: the data flow from raw data to the final file.

Run from the repository root:
    .venv/bin/python submission_model/scripts/make_architecture_diagram.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUTPUT_PATH = "submission_model/architecture.png"
STAGE_COLOURS = {"data": "#dbe8f5", "etl": "#e3f1e0", "model": "#fdf0d5", "inference": "#f6dfe3"}


def draw_box(axes, x, y, width, height, title, lines, stage):
    """One labelled box: a bold title and smaller detail lines below it."""
    axes.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.08",
                                  facecolor=STAGE_COLOURS[stage], edgecolor="#4a4a4a", linewidth=1.1))
    axes.text(x + width / 2, y + height - 0.16, title, ha="center", va="top", fontsize=10.5, weight="bold")
    axes.text(x + width / 2, y + height - 0.52, "\n".join(lines), ha="center", va="top", fontsize=8.3,
              linespacing=1.35)


def draw_arrow(axes, start, end, curve=0.0):
    """An arrow from start to end."""
    axes.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.2,
                                   color="#333333", connectionstyle=f"arc3,rad={curve}"))


figure, axes = plt.subplots(figsize=(15, 6.6))
axes.set_xlim(0, 15)
axes.set_ylim(0, 6.6)
axes.axis("off")

for centre_x, stage_title in [(1.35, "1. Data"), (4.45, "2. ETL"), (8.0, "3. Modelling"), (12.3, "4. Inference")]:
    axes.text(centre_x, 6.5, stage_title, ha="center", va="top", fontsize=12.5, weight="bold", color="#222222")

draw_box(axes, 0.2, 4.2, 2.3, 1.55, "Competition data",
         ["data/Train.csv (2.15 M rows)", "data/Test.csv (281 k rows)", "CSV, standardised values"], "data")
draw_box(axes, 0.2, 2.1, 2.3, 1.55, "ERA5 monthly means",
         ["Copernicus CDS, 1 degree grid", "9 variables, 2002-2018", "netCDF, scripts/fetch_era5.py"], "data")

draw_box(axes, 3.0, 2.0, 2.9, 3.85, "Feature building",
         ["src/tws", "panel.py: cell x month TWS panel", "features.py: anchor, lags, path", "and 500 km neighbour features",
          "era5.py: anomalies at cell centres", "Only months <= t are read.", "No latitude or longitude", "as model features."], "etl")

draw_box(axes, 6.4, 4.75, 3.2, 1.15, "LightGBM (CPU)", ["era5_v5, final_v4, covonly_big,", "dart, lineartree, recursive"], "model")
draw_box(axes, 6.4, 3.35, 3.2, 1.15, "MLP bags (CPU)", ["mlpbag (4 nets), mlpbag8 (8 nets)"], "model")
draw_box(axes, 6.4, 1.95, 3.2, 1.15, "Simple members", ["eof, climatology,", "nb_persist, probe_persist"], "model")
draw_box(axes, 6.4, 0.55, 3.2, 1.15, "Field CNNs (GPU)", ["fieldcnn, fielddirect3"], "model")

draw_box(axes, 10.4, 2.2, 3.0, 3.7, "Blend recipe",
         ["recipe.json, rebuild_final.py", "replay_chain.py: 12 steps", "3 linear steps", "re-anchor patch (980 rows)",
          "2-member solves, 1 scale step", "final step:", "1.1824 blend_nb", "- 0.1824 blend_dart"], "inference")
draw_box(axes, 13.7, 3.3, 1.2, 1.5, "Output", ["final_dart.csv", "ID, Target", "MD5 72e6ab35"], "inference")

draw_arrow(axes, (2.5, 4.95), (3.0, 4.6))
draw_arrow(axes, (2.5, 2.85), (3.0, 3.2))
for model_centre_y in (5.32, 3.92, 2.52, 1.12):
    draw_arrow(axes, (5.9, 3.9), (6.4, model_centre_y))
for model_centre_y in (5.32, 3.92, 2.52):
    draw_arrow(axes, (9.6, model_centre_y), (10.4, 4.3))
draw_arrow(axes, (9.6, 1.3), (10.4, 2.6))
# The feedback path: the recipe hands two earlier blends to the CNN members.
draw_arrow(axes, (11.9, 2.2), (9.6, 0.8), curve=-0.25)
axes.text(12.0, 1.05, "blend_d2_fx and blend_cnn\ngo back to the CNNs", ha="left", va="center", fontsize=8.2,
          style="italic", color="#333333")
draw_arrow(axes, (13.4, 4.05), (13.7, 4.05))

axes.text(7.5, 0.12, "The CNN members read earlier blend files for the rows they do not predict, "
          "so they run inside the blend chain.", ha="center", va="bottom", fontsize=8.3, color="#444444")
figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"wrote {OUTPUT_PATH}")
