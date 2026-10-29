"""Feature analysis on the saved moderate-scenario cohort (uses the locked split from the main run).
Usage: python natto_features.py   -> ../results/features_results.json, features_*.csv, fig5_features.png
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from natto_synth import CLINICAL, LIFESTYLE, DIET, NATTO

warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parent.parent / "results"
rng = np.random.default_rng(2026)

d = pd.read_csv(R / "synthetic_cohort_moderate.csv")
d = d[~d.invalid_hba1c & ~d.no_followup].copy()
man = pd.read_csv(R / "split_manifest.csv")
ALL = CLINICAL + LIFESTYLE + DIET + NATTO
GROUPS = {"glycemic": ["hba1c0", "fpg", "duration"],
          "treatment": ["insulin", "metformin", "sulfonylurea", "dpp4", "vka", "arm"],
          "demographic": ["age", "male", "region_west", "smoking"],
          "body_renal_lipid": ["bmi", "tg", "hdl", "egfr", "hb"],
          "lifestyle": LIFESTYLE, "diet_other": DIET, "natto": NATTO}
assert set(sum(GROUPS.values(), [])) == set(ALL), set(ALL) ^ set(sum(GROUPS.values(), []))
d["tg_hdl"] = d.get("tg_hdl", d.tg / d.hdl)
tr = d.loc[d.index.intersection(man.loc[man.set == "train", "index"])]
te = d.loc[d.index.intersection(man.loc[man.set == "test", "index"])]
if len(tr) + len(te) != len(d):  # manifest index not aligned: fall back to stratified 80/20 split
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(d, test_size=0.2, stratify=d.response, random_state=2026)
ytr, yte = tr.response.values, te.response.values
print("train", len(tr), "test", len(te))


def model(C=0.1):
    return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                     ("m", LogisticRegression(penalty="l2", C=C, max_iter=2000))])


def fit_auc(cols, B=0):
    m = model().fit(tr[cols], ytr)
    p = m.predict_proba(te[cols])[:, 1]
    return m, p, roc_auc_score(yte, p)


def boot_diff(p1, p0, B=1000):
    n = len(yte); out = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if yte[i].min() == yte[i].max():
            continue
        out.append(roc_auc_score(yte[i], p1[i]) - roc_auc_score(yte[i], p0[i]))
    return np.percentile(out, [2.5, 97.5])


res = {}

# 1. univariate screen (train only)
uni = []
mi = mutual_info_classif(SimpleImputer(strategy="median").fit_transform(tr[ALL]), ytr, random_state=0)
for f, m_ in zip(ALL, mi):
    x = tr[f]; ok = x.notna()
    a = roc_auc_score(ytr[ok], x[ok])
    uni.append(dict(feature=f, univariate_auroc=max(a, 1 - a), direction="+" if a >= .5 else "-",
                    mutual_info=m_, missing_pct=100 * x.isna().mean()))
uni = pd.DataFrame(uni).sort_values("univariate_auroc", ascending=False)
uni.to_csv(R / "features_univariate.csv", index=False)

# 2. collinearity: pairwise correlations and VIF
Z = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(tr[ALL]), columns=ALL)
C = Z.corr()
pairs = [(a, b, C.loc[a, b]) for i, a in enumerate(ALL) for b in ALL[i + 1:] if abs(C.loc[a, b]) >= 0.4]
Zs = (Z - Z.mean()) / Z.std()
def _vif(i, f):
    X = np.delete(Zs.values, i, 1); y = Zs[f].values
    ssr = np.sum((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2)
    return np.sum(y ** 2) / ssr


vif = {f: _vif(i, f) for i, f in enumerate(ALL)}
res["corr_pairs_ge_0.4"] = [dict(a=a, b=b, r=round(r, 2)) for a, b, r in sorted(pairs, key=lambda t: -abs(t[2]))]
res["vif"] = {k: round(float(v), 2) for k, v in vif.items()}
res["max_vif"] = max(res["vif"].items(), key=lambda t: t[1])

# 3. group ablation on the test set: drop one group from the full model; add one group to the base
m_full, p_full, a_full = fit_auc(ALL)
res["full_auroc"] = round(a_full, 3)
drop, add = [], []
base_cols = GROUPS["glycemic"]
m_b, p_b, a_b = fit_auc(base_cols)
for g, cols in GROUPS.items():
    keep = [c for c in ALL if c not in cols]
    _, p, a = fit_auc(keep)
    lo, hi = boot_diff(p_full, p)
    drop.append(dict(group=g, auroc_without=a, delta_from_full=a_full - a, ci_lo=lo, ci_hi=hi))
    if g != "glycemic":
        _, p, a = fit_auc(base_cols + cols)
        lo, hi = boot_diff(p, p_b)
        add.append(dict(group=g, auroc_with=a, delta_from_glycemic=a - a_b, ci_lo=lo, ci_hi=hi))
drop = pd.DataFrame(drop).sort_values("delta_from_full", ascending=False); add = pd.DataFrame(add).sort_values("delta_from_glycemic", ascending=False)
drop.to_csv(R / "features_group_drop.csv", index=False); add.to_csv(R / "features_group_add.csv", index=False)
res["glycemic_only_auroc"] = round(a_b, 3)

# 4. permutation importance on the test set (30 repeats), grouped for natto
perm = []
for f in ALL:
    drops = []
    for _ in range(30):
        Xp = te[ALL].copy(); Xp[f] = rng.permutation(Xp[f].values)
        drops.append(a_full - roc_auc_score(yte, m_full.predict_proba(Xp)[:, 1]))
    perm.append(dict(feature=f, perm_drop=np.mean(drops), sd=np.std(drops)))
perm = pd.DataFrame(perm).sort_values("perm_drop", ascending=False)
perm.to_csv(R / "features_permutation.csv", index=False)

# 5. selection stability: L1 logistic over 100 bootstrap resamples of the training set
sel = pd.Series(0.0, index=ALL)
for b in range(100):
    i = rng.integers(0, len(tr), len(tr))
    mm = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                   ("m", LogisticRegression(penalty="l1", solver="liblinear", C=0.02, max_iter=500))])
    mm.fit(tr[ALL].iloc[i], ytr[i])
    sel += (np.abs(mm[-1].coef_[0]) > 1e-8)
sel = (sel / 100).sort_values(ascending=False)
sel.to_csv(R / "features_stability.csv", header=["selection_freq"])

# 6. natto dose-response: observed rate and adjusted (covariate-standardised) predicted rate by category
cats = {0: "never", 1: "<1/week", 2: "1-3/week", 3: ">=4/week"}
obs = d.groupby("natto_cat").response.agg(["mean", "size"])
Xall = d[ALL].copy()
m_all = model().fit(tr[ALL], ytr)
dose = []
for k in cats:
    X2 = Xall.copy(); X2["natto_cat"] = k
    dose.append(dict(cat=cats[k], observed=float(obs.loc[k, "mean"]), n=int(obs.loc[k, "size"]),
                     adjusted=float(m_all.predict_proba(X2)[:, 1].mean())))
dose = pd.DataFrame(dose); dose.to_csv(R / "features_dose_response.csv", index=False)

# 7. interaction screen: natto x each other feature added to the clinical+natto logistic model (test AUROC gain, LR on train)
cn = CLINICAL + NATTO
_, p_cn, a_cn = fit_auc(cn)
inter = []
for f in CLINICAL:
    t2, e2 = tr.copy(), te.copy()
    for D in (t2, e2):
        D["x_int"] = D.natto_cat.fillna(D.natto_cat.median()) * ((D[f] - tr[f].mean()) / tr[f].std())
    m2 = model().fit(t2[cn + ["x_int"]], ytr)
    a2 = roc_auc_score(yte, m2.predict_proba(e2[cn + ["x_int"]])[:, 1])
    inter.append(dict(feature=f, delta_auroc=a2 - a_cn, coef=m2[-1].coef_[0][-1]))
inter = pd.DataFrame(inter).sort_values("coef", key=abs, ascending=False)
inter.to_csv(R / "features_interactions.csv", index=False)

res.update(dict(
    top_univariate=uni.head(5).to_dict("records"), natto_univariate=uni[uni.feature == "natto_cat"].to_dict("records"),
    drop=drop.round(4).to_dict("records"), add=add.round(4).to_dict("records"),
    perm_top=perm.head(6).round(4).to_dict("records"), perm_natto=perm[perm.feature == "natto_cat"].round(4).to_dict("records"),
    stability_top=sel.head(8).round(2).to_dict(), stability_natto=float(sel["natto_cat"]),
    dose=dose.round(3).to_dict("records"), interactions_top=inter.head(4).round(4).to_dict("records"),
    n_train=len(tr), n_test=len(te)))
json.dump(res, open(R / "features_results.json", "w"), indent=1, default=float)

# figure
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
a = ax[0, 0]; q = perm.head(12).iloc[::-1]
a.barh(q.feature, q.perm_drop, xerr=q.sd, color=["#c0504d" if f == "natto_cat" else "#4f81bd" for f in q.feature])
a.set_title("A. Permutation importance (test AUROC drop)"); a.set_xlabel("AUROC drop")
a = ax[0, 1]; q = drop.iloc[::-1]
a.errorbar(q.delta_from_full, range(len(q)), xerr=[q.delta_from_full - q.ci_lo, q.ci_hi - q.delta_from_full], fmt="o", color="#4f81bd")
a.set_yticks(range(len(q))); a.set_yticklabels(q.group); a.axvline(0, color="k", lw=.7)
a.set_title("B. Drop-one-group (AUROC lost vs full model)"); a.set_xlabel("AUROC lost (95% CI)")
a = ax[1, 0]; q = sel.head(12).iloc[::-1]
a.barh(q.index, q.values, color=["#c0504d" if f == "natto_cat" else "#4f81bd" for f in q.index])
a.set_title("C. L1 selection frequency (100 bootstraps)"); a.set_xlim(0, 1)
a = ax[1, 1]
a.plot(dose.cat, dose.observed, "o-", label="observed"); a.plot(dose.cat, dose.adjusted, "s--", label="covariate-standardised")
a.set_title("D. Natto frequency and response rate"); a.set_ylabel("P(response)"); a.legend()
plt.tight_layout(); plt.savefig(R / "fig5_features.png", dpi=600)
print(json.dumps({k: res[k] for k in ["full_auroc", "glycemic_only_auroc", "max_vif", "stability_natto"]}))
print(drop.round(3).to_string()); print(add.round(3).to_string()); print(perm.head(6).round(3).to_string())
print(uni.head(8).round(3).to_string()); print(sel.head(10).round(2).to_string()); print(dose.round(3).to_string())
print(inter.head(4).round(3).to_string()); print(res["corr_pairs_ge_0.4"][:6])
