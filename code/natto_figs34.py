"""Re-draw Figures 3 and 4 at 600 dpi (PNG and PDF) from the saved cohort, split and tuned parameters.
Refits only the two penalized logistic models used in the figures (the selected model) and checks the
calibration slopes and SHAP ranking against main_results.json before saving.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from natto_synth import FEATURE_SETS

R = Path(__file__).resolve().parent.parent / "results"
res = json.load(open(R / "main_results.json"))
d = pd.read_csv(R / "synthetic_cohort_moderate.csv")
man = pd.read_csv(R / "split_manifest.csv")
tr = man.loc[man.set == "train", "index"].values; te = man.loc[man.set == "test", "index"].values
ytr, yte = d.response.values[tr], d.response.values[te]
assert res["selected_model"] == "logistic"


def fit(fs):
    C = res["best_params"][f"{fs}|logistic"]["C"]
    m = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=2000, C=C))])
    cols = FEATURE_SETS[fs]
    return m.fit(d.iloc[tr][cols], ytr), cols


def slope(y, p):
    from sklearn.linear_model import LogisticRegression as LR
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return float(LR(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y).coef_[0][0])


fits = {fs: fit(fs) for fs in ("clinical", "clinical+natto")}
preds = {fs: m.predict_proba(d.iloc[te][cols])[:, 1] for fs, (m, cols) in fits.items()}
for fs, key in (("clinical", "cal_clinical"), ("clinical+natto", "cal_natto")):
    s = slope(yte, preds[fs])
    print(fs, "slope refit", round(s, 3), "saved", round(res[key]["slope"], 3))
    assert abs(s - res[key]["slope"]) < 0.02, (fs, s, res[key]["slope"])

# Figure 3
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
for fs, c in [("clinical", "#888"), ("clinical+natto", "#1f6f8b")]:
    pp = preds[fs]
    q = pd.qcut(pp, 10, duplicates="drop")
    g = pd.DataFrame({"p": pp, "y": yte, "q": q}).groupby("q", observed=True).mean()
    ax[0].plot(g.p, g.y, "o-", color=c, label=fs, ms=4)
ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].set_xlim(0, 0.9); ax[0].set_ylim(0, 0.9)
ax[0].set_xlabel("Predicted probability"); ax[0].set_ylabel("Observed frequency")
ax[0].set_title("Calibration (deciles)"); ax[0].legend(frameon=False)
th = np.arange(0.20, 0.61, 0.02); n = len(yte); prev = yte.mean()
nb = lambda pp: [np.sum((pp >= t) & (yte == 1)) / n - np.sum((pp >= t) & (yte == 0)) / n * t / (1 - t) for t in th]
ax[1].plot(th, [prev - (1 - prev) * t / (1 - t) for t in th], color="#bbb", label="treat all")
ax[1].axhline(0, color="k", lw=0.8, label="treat none")
for fs, c in [("clinical", "#888"), ("clinical+natto", "#1f6f8b")]:
    ax[1].plot(th, nb(preds[fs]), color=c, label=fs)
ax[1].set_xlabel("Threshold probability"); ax[1].set_ylabel("Net benefit")
ax[1].set_title("Decision curve"); ax[1].legend(frameon=False, fontsize=8); ax[1].set_ylim(-0.05, prev)
plt.tight_layout(); plt.savefig(R / "fig3_calibration_dca.png", dpi=600); plt.savefig(R / "fig3_calibration_dca.pdf"); plt.close()

# Figure 4
mdl, cols = fits["clinical+natto"]
Xte, Xtr = d.iloc[te][cols], d.iloc[tr][cols]
imp = mdl.named_steps["imp"]
names = cols + [f"{cols[i]}_missing" for i in imp.indicator_.features_]
Xte_t = pd.DataFrame(imp.transform(Xte), columns=names)
bg = pd.DataFrame(imp.transform(Xtr.sample(100, random_state=0)), columns=names)
sc_ = mdl.named_steps["sc"]
Xte_t = pd.DataFrame(sc_.transform(Xte_t), columns=names); bg = pd.DataFrame(sc_.transform(bg), columns=names)
sv = shap.LinearExplainer(mdl.named_steps["clf"], bg).shap_values(Xte_t)
sv = sv[1] if isinstance(sv, list) else sv
top = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)
print(top.head(3).round(3).to_dict(), "saved", {k: round(v, 3) for k, v in list(res["shap_top"].items())[:3]})
assert list(top.index[:3]) == list(res["shap_top"])[:3] and abs(top.iloc[0] - list(res["shap_top"].values())[0]) < 0.01
plt.figure(figsize=(7, 4.5))
shap.summary_plot(sv, Xte_t, max_display=12, show=False, plot_size=None)
plt.tight_layout(); plt.savefig(R / "fig4_shap.png", dpi=600); plt.savefig(R / "fig4_shap.pdf"); plt.close()
print("figures 3 and 4 saved at 600 dpi")
