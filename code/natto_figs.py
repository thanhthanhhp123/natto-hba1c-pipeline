"""Figure 1 (participant flow) and Figure 2 (DAG), drawn from main_results.json."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "results"
r = json.load(open(OUT / "main_results.json"))
f = r["flow"]


def box(ax, x, y, w, h, text, fc="#eef3f6"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec="#33566b", lw=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=9)


def arrow(ax, a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="#33566b", lw=1.2))


fig, ax = plt.subplots(figsize=(7, 6)); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
box(ax, 4, 9, 5, 1, f"Synthetic records generated\n(n = {f['enrolled']:,})")
box(ax, 4, 6.8, 5, 1, f"Records with valid HbA1c\n(n = {f['after_invalid']:,})")
box(ax, 4, 4.6, 5, 1, f"Analysis set with follow-up in window\n(n = {f['analysed']:,})")
box(ax, 2.2, 2.2, 3.6, 1.2, f"Training set\n(n = {r['n_train']:,})")
box(ax, 6.4, 2.2, 3.6, 1.2, f"Locked test set\n(n = {r['n_test']:,})")
box(ax, 8.3, 7.9, 3.0, 1.3, f"Excluded: conditions that\ninvalidate HbA1c\n(n = {f['invalid_hba1c']})", "#fbeee6")
box(ax, 8.3, 5.7, 3.0, 1.3, f"Excluded: no HbA1c in\nfollow-up window\n(n = {f['no_followup']})", "#fbeee6")
arrow(ax, (4, 8.5), (4, 7.3)); arrow(ax, (4, 6.3), (4, 5.1))
arrow(ax, (4, 4.1), (2.2, 2.8)); arrow(ax, (4, 4.1), (6.4, 2.8))
arrow(ax, (6.5, 7.9), (6.8, 7.9)); arrow(ax, (4, 7.9), (6.8, 7.9))
arrow(ax, (4, 5.7), (6.8, 5.7))
ax.text(4, 0.7, f"Natto frequency missing in {f['natto_missing']} records (kept as missing in the\n"
        "prediction models, complete-case in the causal analyses).", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "fig1_flow.png", dpi=600); plt.close()


fig, ax = plt.subplots(figsize=(7, 5.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
box(ax, 2.5, 6.7, 3.8, 1.5, "Confounders\n(age, region, baseline HbA1c,\nactivity, diet pattern)", "#fff4d6")
box(ax, 7.5, 6.7, 3.8, 1.5, "Vitamin K antagonist\ntherapy", "#fff4d6")
box(ax, 2.5, 3.9, 3.0, 1.2, "Natto frequency")
box(ax, 7.5, 3.9, 3.0, 1.2, "HbA1c change\nat 12 months")
box(ax, 5.0, 1.5, 3.6, 1.4, "Mediators\n(substitution, MK-7, γ-PGA)")
arrow(ax, (2.5, 5.95), (2.5, 4.5)); arrow(ax, (7.5, 5.95), (7.5, 4.5))
arrow(ax, (4.2, 6.0), (6.3, 4.5)); arrow(ax, (5.8, 6.0), (3.7, 4.5))
arrow(ax, (4.0, 3.9), (6.0, 3.9))
arrow(ax, (3.2, 3.3), (4.0, 2.2)); arrow(ax, (6.0, 2.2), (6.8, 3.3))
ax.text(5, 0.25, "Adjusted: confounders and vitamin K antagonist therapy. Mediators are not adjusted.",
        ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "fig2_dag.png", dpi=600); plt.close()
print("figs ok")
