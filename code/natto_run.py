"""Analysis pipeline (Sections 2.8-2.15). Usage:
    python natto_run.py main        # full analysis on one synthetic cohort
    python natto_run.py scenarios   # repeated seeds over declared scenarios
Outputs go to ../results (JSON, CSV, PNG).
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from types import SimpleNamespace
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools.tools import add_constant
sm = SimpleNamespace(GLM=GLM, families=families, OLS=OLS, WLS=WLS, add_constant=add_constant)
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from natto_synth import FEATURE_SETS, SCENARIOS, CLINICAL, generate

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)
MAIN_SEED, N_ENROLLED = 2026, 9000
CONFOUNDERS = ["age", "male", "bmi", "duration", "hba1c0", "egfr", "insulin", "metformin",
               "sulfonylurea", "dpp4", "vka", "smoking", "arm", "region_west",
               "activity", "energy", "rice", "veg", "fish", "other_soy"]


def pipe(clf, scale=False):
    steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("sc", StandardScaler()))
    return Pipeline(steps + [("clf", clf)])


def model_grid(small=False):
    g = {
        "logistic": (pipe(LogisticRegression(max_iter=2000), True), {"clf__C": [0.01, 0.1, 1.0]}),
        "random_forest": (pipe(RandomForestClassifier(n_estimators=300, n_jobs=4, random_state=0)),
                          {"clf__max_depth": [4, 8], "clf__min_samples_leaf": [20, 50]}),
        "xgboost": (pipe(XGBClassifier(subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                                       n_jobs=4, random_state=0, eval_metric="logloss")),
                    {"clf__n_estimators": [150, 300], "clf__max_depth": [2, 3],
                     "clf__learning_rate": [0.03] if small else [0.03, 0.08]}),
        "lightgbm": (pipe(LGBMClassifier(min_child_samples=40, subsample=0.8, subsample_freq=1,
                                         colsample_bytree=0.8, n_jobs=4, random_state=0,
                                         verbose=-1)),
                     {"clf__n_estimators": [150, 300], "clf__num_leaves": [7, 15],
                      "clf__learning_rate": [0.03, 0.08]}),
    }
    return g


def fit_tuned(name, X, y, small=False, seed=0):
    est, grid = model_grid(small)[name]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    gs = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=1, refit=True)
    gs.fit(X, y)
    return gs


def cal_stats(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    inter = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
    return float(inter), float(slope)


def boot_ci(fn, n, B=1000, seed=0):
    r = np.random.default_rng(seed)
    v = np.array([fn(r.integers(0, n, n)) for _ in range(B)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def auc_ci(y, p, B=1000):
    ci = boot_ci(lambda i: roc_auc_score(y[i], p[i]) if 0 < y[i].sum() < len(i) else np.nan, len(y), B)
    return float(roc_auc_score(y, p)), ci


def smd(a, b):
    a, b = a.dropna(), b.dropna()
    s = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def weighted_smd(x, g, w):
    m1 = np.average(x[g == 1], weights=w[g == 1]); m0 = np.average(x[g == 0], weights=w[g == 0])
    v1 = np.average((x[g == 1] - m1) ** 2, weights=w[g == 1])
    v0 = np.average((x[g == 0] - m0) ** 2, weights=w[g == 0])
    s = np.sqrt((v1 + v0) / 2)
    return float((m1 - m0) / s) if s > 0 else 0.0


def prep_causal(d):
    d = d.dropna(subset=["natto_cat"]).copy()
    X = d[CONFOUNDERS].copy()
    X = X.fillna(X.median())
    return d, X


def causal(d, excl_vka=False, trim=True, B=200, seed=0):
    """High (>=4/wk) vs never/low (<1/wk) contrast; the middle category is excluded here only."""
    d = d[d["natto_cat"].isin([0, 1, 2, 3])]
    if excl_vka:
        d = d[d["vka"] == 0]
    d, X = prep_causal(d)
    keep = d["natto_cat"].isin([0, 1, 3]).values
    d, X = d[keep], X[keep]
    g = (d["natto_cat"] == 3).astype(int).values
    y = d["response"].values; dl = d["delta"].values

    def one(idx, report=False):
        Xi, gi, yi, di = X.values[idx], g[idx], y[idx], dl[idx]
        Z = StandardScaler().fit_transform(Xi)
        ps = LogisticRegression(C=1e3, max_iter=2000).fit(Z, gi).predict_proba(Z)[:, 1]
        if trim:
            lo, hi = np.percentile(ps, [1, 99]); ok = (ps >= lo) & (ps <= hi)
        else:
            ok = np.ones_like(ps, bool)
        w = np.where(gi == 1, gi.mean() / ps, (1 - gi.mean()) / (1 - ps))[ok]
        gi2, yi2, di2, Zi = gi[ok], yi[ok], di[ok], Z[ok]
        crude = sm.GLM(yi, sm.add_constant(gi), family=sm.families.Binomial()).fit().params[1]
        ipw_or = sm.GLM(yi2, sm.add_constant(gi2), family=sm.families.Binomial(),
                        freq_weights=w).fit().params[1]
        ipw_d = sm.WLS(di2, sm.add_constant(gi2), weights=w).fit().params[1]
        A = sm.add_constant(np.column_stack([gi, Zi_full(Z)]))
        adj_or = sm.GLM(yi, A, family=sm.families.Binomial()).fit().params[1]
        adj_d = sm.OLS(di, A).fit().params[1]
        res = dict(crude_or=crude, adj_or=adj_or, adj_delta=adj_d, ipw_or=ipw_or, ipw_delta=ipw_d)
        if report:
            res["ps"], res["ok"], res["w"], res["Xi"], res["gi"] = ps, ok, w, Xi, gi
        return res

    def Zi_full(Z):
        return Z

    base = one(np.arange(len(g)), report=True)
    ps, ok, w, gi = base.pop("ps"), base.pop("ok"), base.pop("w"), base.pop("gi")
    Xi = base.pop("Xi")
    pre = {c: abs(smd(d.loc[g == 1, c] if c in d else pd.Series(Xi[g == 1, j]),
                      d.loc[g == 0, c] if c in d else pd.Series(Xi[g == 0, j])))
           for j, c in enumerate(CONFOUNDERS)}
    post = {c: abs(weighted_smd(Xi[ok][:, j], gi[ok], w)) for j, c in enumerate(CONFOUNDERS)}
    r = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        idx = r.integers(0, len(g), len(g))
        try:
            boots.append(one(idx))
        except Exception:
            pass
    bd = pd.DataFrame(boots)
    out = {}
    for k, v in base.items():
        lo, hi = np.percentile(bd[k], [2.5, 97.5])
        e = lambda x: float(np.exp(x)) if k.endswith("or") else float(x)
        out[k] = dict(est=e(v), lo=e(lo), hi=e(hi))
    out.update(n=int(len(g)), n_high=int(g.sum()), n_low=int((1 - g).sum()),
               n_trimmed=int((~ok).sum()), max_smd_before=float(max(pre.values())),
               max_smd_after=float(max(post.values())),
               ps_min=float(ps.min()), ps_max=float(ps.max()))
    return out


def run_main():
    raw = generate(N_ENROLLED, "moderate", MAIN_SEED)
    flow = dict(enrolled=len(raw))
    d = raw[~raw.invalid_hba1c]; flow["invalid_hba1c"] = int(raw.invalid_hba1c.sum())
    flow["after_invalid"] = len(d)
    d = d[~d.no_followup].reset_index(drop=True)
    flow["no_followup"] = flow["after_invalid"] - len(d); flow["analysed"] = len(d)
    flow["natto_missing"] = int(d.natto_cat.isna().sum())
    flow["vka_users"] = int(d.vka.sum())
    res = dict(flow=flow, prevalence=float(d.response.mean()), seed=MAIN_SEED)
    d.to_csv(OUT / "synthetic_cohort_moderate.csv", index=False)

    # Table 1
    lab = d["natto_cat"].map({0: "never/low", 1: "never/low", 2: "moderate", 3: "high"})
    t1 = []
    for c, nm in [("age", "Age, years"), ("male", "Male sex, %"), ("bmi", "BMI, kg/m2"),
                  ("duration", "Diabetes duration, years"), ("hba1c0", "Baseline HbA1c, % (NGSP)"),
                  ("egfr", "eGFR, mL/min/1.73 m2"), ("insulin", "Insulin use, %"),
                  ("vka", "Vitamin K antagonist use, %"), ("region_west", "Western Japan, %"),
                  ("rice", "Rice, servings/day"), ("veg", "Vegetables, servings/day"),
                  ("activity", "Physical activity, MET-h/week")]:
        row = {"Variable": nm}
        pct = c in ("male", "insulin", "vka", "region_west")
        for g in ["never/low", "moderate", "high"]:
            s = d.loc[lab == g, c].dropna()
            row[g] = f"{100*s.mean():.1f}" if pct else f"{s.mean():.1f} ({s.std():.1f})"
        row["SMD (high vs never/low)"] = f"{smd(d.loc[lab=='high', c], d.loc[lab=='never/low', c]):.2f}"
        t1.append(row)
    t1.insert(0, {"Variable": "n", "never/low": int((lab == 'never/low').sum()),
                  "moderate": int((lab == 'moderate').sum()), "high": int((lab == 'high').sum()),
                  "SMD (high vs never/low)": ""})
    t1.append({"Variable": "Responders (HbA1c fall >=0.5), %",
               **{g: f"{100*d.loc[lab==g,'response'].mean():.1f}" for g in ["never/low", "moderate", "high"]},
               "SMD (high vs never/low)": ""})
    res["table1"] = t1
    res["natto_missing_pct"] = float(100 * d.natto_cat.isna().mean())

    # locked split
    tr, te = train_test_split(np.arange(len(d)), test_size=0.2, stratify=d.response, random_state=MAIN_SEED)
    pd.DataFrame({"index": np.r_[tr, te], "set": ["train"] * len(tr) + ["test"] * len(te)}
                 ).to_csv(OUT / "split_manifest.csv", index=False)
    ytr, yte = d.response.values[tr], d.response.values[te]
    res["n_train"], res["n_test"] = len(tr), len(te)

    # models x feature sets
    import joblib
    cache = OUT / "fits_cache.joblib"
    if cache.exists():
        fits, preds, cvauc, params = joblib.load(cache)
    else:
        fits, preds, cvauc, params = {}, {}, {}, {}
        for fs, cols in FEATURE_SETS.items():
            for name in model_grid():
                gs = fit_tuned(name, d.iloc[tr][cols], ytr, seed=MAIN_SEED)
                fits[(fs, name)] = gs.best_estimator_
                preds[(fs, name)] = gs.best_estimator_.predict_proba(d.iloc[te][cols])[:, 1]
                cvauc[(fs, name)] = float(gs.best_score_)
                params[f"{fs}|{name}"] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
                print(fs, name, round(gs.best_score_, 4), flush=True)
        joblib.dump((fits, preds, cvauc, params), cache)
    res["best_params"] = params
    res["cv_auc"] = {f"{a}|{b}": v for (a, b), v in cvauc.items()}

    t3 = []
    for name in model_grid():
        p = preds[("clinical+natto", name)]
        a, ci = auc_ci(yte, p); i, s = cal_stats(yte, p)
        t3.append(dict(Model=name, AUROC=f"{a:.3f} ({ci[0]:.3f}-{ci[1]:.3f})",
                       AUPRC=f"{average_precision_score(yte, p):.3f}",
                       Brier=f"{brier_score_loss(yte, p):.3f}",
                       Cal_intercept=f"{i:.2f}", Cal_slope=f"{s:.2f}",
                       CV_AUROC=f"{cvauc[('clinical+natto', name)]:.3f}"))
    res["table3"] = t3
    res["prev_test"] = float(yte.mean())

    # algorithm chosen on training-fold AUROC only (clinical+natto)
    best = max(model_grid(), key=lambda m: cvauc[("clinical+natto", m)])
    res["selected_model"] = best
    t4, base_p = [], preds[("clinical", best)]
    rng = np.random.default_rng(1)
    B = 1000
    idxs = [rng.integers(0, len(yte), len(yte)) for _ in range(B)]
    for fs in FEATURE_SETS:
        p = preds[(fs, best)]
        a, ci = auc_ci(yte, p)
        dA = [roc_auc_score(yte[i], p[i]) - roc_auc_score(yte[i], base_p[i]) for i in idxs]
        dB = [brier_score_loss(yte[i], p[i]) - brier_score_loss(yte[i], base_p[i]) for i in idxs]
        f = lambda v: f"{np.mean(v):+.3f} ({np.percentile(v,2.5):+.3f} to {np.percentile(v,97.5):+.3f})"
        t4.append(dict(Feature_set=fs, AUROC=f"{a:.3f} ({ci[0]:.3f}-{ci[1]:.3f})",
                       dAUROC=("reference" if fs == "clinical" else f(dA)),
                       dBrier=("reference" if fs == "clinical" else f(dB)),
                       _dA=float(np.mean(dA)), _lo=float(np.percentile(dA, 2.5)),
                       _hi=float(np.percentile(dA, 97.5))))
    res["table4"] = t4

    # repeated-split sensitivity for the natto increment (hyperparameters fixed)
    est_c = clone(fits[("clinical", best)]); est_n = clone(fits[("clinical+natto", best)])
    inc = []
    for s in range(20):
        tr2, te2 = train_test_split(np.arange(len(d)), test_size=0.2, stratify=d.response, random_state=100 + s)
        pc = clone(est_c).fit(d.iloc[tr2][FEATURE_SETS["clinical"]], d.response.values[tr2])
        pn = clone(est_n).fit(d.iloc[tr2][FEATURE_SETS["clinical+natto"]], d.response.values[tr2])
        yy = d.response.values[te2]
        inc.append(roc_auc_score(yy, pn.predict_proba(d.iloc[te2][FEATURE_SETS["clinical+natto"]])[:, 1])
                   - roc_auc_score(yy, pc.predict_proba(d.iloc[te2][FEATURE_SETS["clinical"]])[:, 1]))
    res["repeat_split"] = dict(mean=float(np.mean(inc)), sd=float(np.std(inc, ddof=1)),
                               min=float(np.min(inc)), max=float(np.max(inc)))

    # subgroups, selected model clinical+natto
    p = preds[("clinical+natto", best)]; te_df = d.iloc[te].reset_index(drop=True)
    subs = {"Age <60": te_df.age < 60, "Age >=60": te_df.age >= 60, "Male": te_df.male == 1,
            "Female": te_df.male == 0, "Baseline HbA1c <8.0": te_df.hba1c0 < 8,
            "Baseline HbA1c >=8.0": te_df.hba1c0 >= 8, "Insulin": te_df.insulin == 1,
            "No insulin": te_df.insulin == 0}
    t5 = []
    for k, m in subs.items():
        m = m.values; a, ci = auc_ci(yte[m], p[m], B=500)
        t5.append(dict(Subgroup=k, n=int(m.sum()), AUROC=f"{a:.3f} ({ci[0]:.3f}-{ci[1]:.3f})"))
    res["table5"] = t5
    inter = {}
    for nm, col in [("age>=60", (d.age >= 60).astype(int)), ("male", d.male),
                    ("hba1c0>=8", (d.hba1c0 >= 8).astype(int)), ("insulin", d.insulin)]:
        dd = d.dropna(subset=["natto_cat"])
        x = (dd.natto_cat == 3).astype(int); z = col.loc[dd.index]
        A = sm.add_constant(np.column_stack([x, z, x * z, (dd[["hba1c0"]].values.ravel())]))
        fit = sm.GLM(dd.response.values, A, family=sm.families.Binomial()).fit()
        inter[nm] = float(np.asarray(fit.pvalues)[3])
    res["interaction_p"] = inter

    # calibration and decision curve (selected model, clinical+natto and clinical)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for fs, c in [("clinical", "#888"), ("clinical+natto", "#1f6f8b")]:
        pp = preds[(fs, best)]
        q = pd.qcut(pp, 10, duplicates="drop")
        g = pd.DataFrame({"p": pp, "y": yte, "q": q}).groupby("q", observed=True).mean()
        ax[0].plot(g.p, g.y, "o-", color=c, label=fs, ms=4)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].set_xlim(0, 0.9); ax[0].set_ylim(0, 0.9)
    ax[0].set_xlabel("Predicted probability"); ax[0].set_ylabel("Observed frequency")
    ax[0].set_title("Calibration (deciles)"); ax[0].legend(frameon=False)
    th = np.arange(0.20, 0.61, 0.02); n = len(yte); prev = yte.mean()
    def nb(pp):
        return [np.sum((pp >= t) & (yte == 1)) / n - np.sum((pp >= t) & (yte == 0)) / n * t / (1 - t) for t in th]
    ax[1].plot(th, [prev - (1 - prev) * t / (1 - t) for t in th], color="#bbb", label="treat all")
    ax[1].axhline(0, color="k", lw=0.8, label="treat none")
    for fs, c in [("clinical", "#888"), ("clinical+natto", "#1f6f8b")]:
        ax[1].plot(th, nb(preds[(fs, best)]), color=c, label=fs)
    ax[1].set_xlabel("Threshold probability"); ax[1].set_ylabel("Net benefit")
    ax[1].set_title("Decision curve"); ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_ylim(-0.05, prev)
    plt.tight_layout(); plt.savefig(OUT / "fig3_calibration_dca.png", dpi=600); plt.close()
    nbn, nbc = np.array(nb(preds[("clinical+natto", best)])), np.array(nb(preds[("clinical", best)]))
    nba = np.array([prev - (1 - prev) * t / (1 - t) for t in th])
    res["dca"] = dict(lo=float(th[0]), hi=float(th[-1]), mean_nb_gain_vs_clinical=float(np.mean(nbn - nbc)),
                      mean_nb_gain_vs_all=float(np.mean(nbn - nba)),
                      share_thresholds_above_all=float(np.mean(nbn > nba)))
    ic, sc = cal_stats(yte, preds[("clinical", best)]); inn, sn = cal_stats(yte, preds[("clinical+natto", best)])
    res["cal_clinical"], res["cal_natto"] = dict(intercept=ic, slope=sc), dict(intercept=inn, slope=sn)

    # SHAP on selected model, clinical+natto, held-out sample
    import shap
    mdl = fits[("clinical+natto", best)]
    cols = FEATURE_SETS["clinical+natto"]
    Xte = d.iloc[te][cols]; Xtr = d.iloc[tr][cols]
    imp = mdl.named_steps["imp"]
    names = cols + [f"{cols[i]}_missing" for i in imp.indicator_.features_]
    Xte_t = pd.DataFrame(imp.transform(Xte), columns=names)
    bg = pd.DataFrame(imp.transform(Xtr.sample(100, random_state=0)), columns=names)
    clf = mdl.named_steps["clf"]
    if best == "logistic":
        sc_ = mdl.named_steps["sc"]; Xte_t = pd.DataFrame(sc_.transform(Xte_t), columns=names)
        bg = pd.DataFrame(sc_.transform(bg), columns=names)
        ex = shap.LinearExplainer(clf, bg)
    else:
        ex = shap.TreeExplainer(clf, data=bg, feature_perturbation="interventional", model_output="raw")
    sv = ex.shap_values(Xte_t)
    sv = sv[1] if isinstance(sv, list) else sv
    mean_abs = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)
    res["shap_top"] = {k: float(v) for k, v in mean_abs.head(10).items()}
    res["shap_rank_natto"] = int(list(mean_abs.index).index("natto_cat") + 1)
    res["shap_natto_share"] = float(mean_abs["natto_cat"] / mean_abs.sum())
    # fold stability: rank of natto in 5 bootstrap halves of the test set
    r2 = np.random.default_rng(3); rk = []
    for _ in range(20):
        ii = r2.choice(len(Xte_t), len(Xte_t) // 2, replace=False)
        m = pd.Series(np.abs(sv[ii]).mean(0), index=names).sort_values(ascending=False)
        rk.append(int(list(m.index).index("natto_cat") + 1))
    res["shap_natto_rank_range"] = [int(min(rk)), int(max(rk))]
    plt.figure(figsize=(7, 4.5))
    shap.summary_plot(sv, Xte_t, max_display=12, show=False, plot_size=None)
    plt.tight_layout(); plt.savefig(OUT / "fig4_shap.png", dpi=600); plt.close()

    # causal analyses
    res["causal"] = causal(d, B=200)
    res["causal_noVKA"] = causal(d, excl_vka=True, B=100)
    res["causal_notrim"] = causal(d, trim=False, B=100)
    dd = d.dropna(subset=["natto_cat"]); Xc = dd[CONFOUNDERS].fillna(dd[CONFOUNDERS].median())
    Z = StandardScaler().fit_transform(Xc)
    for nm, cond in [("moderate", 2), ("high", 3)]:
        sel = dd.natto_cat.isin([0, 1, cond]).values
        gg = (dd.natto_cat[sel] == cond).astype(int).values
        A = sm.add_constant(np.column_stack([gg, Z[sel]]))
        f = sm.GLM(dd.response.values[sel], A, family=sm.families.Binomial()).fit()
        ci = f.conf_int()[1]
        res[f"adj_or_{nm}"] = dict(est=float(np.exp(f.params[1])), lo=float(np.exp(ci[0])), hi=float(np.exp(ci[1])))
    res["truth"] = dict(b_high=SCENARIOS["moderate"]["b_high"])

    # external-style robustness: shifted population (not external validation)
    sh = generate(4000, "moderate", MAIN_SEED + 7, shifted=True)
    sh = sh[(~sh.invalid_hba1c) & (~sh.no_followup)]
    shift = {}
    for fs in ["clinical", "clinical+natto"]:
        pp = fits[(fs, best)].predict_proba(sh[FEATURE_SETS[fs]])[:, 1]
        a, ci = auc_ci(sh.response.values, pp, B=500); i, s = cal_stats(sh.response.values, pp)
        shift[fs] = dict(auc=a, lo=ci[0], hi=ci[1], intercept=i, slope=s,
                         brier=float(brier_score_loss(sh.response.values, pp)))
    res["shifted"] = dict(n=len(sh), prevalence=float(sh.response.mean()), **shift)
    json.dump(res, open(OUT / "main_results.json", "w"), indent=1, default=str)
    print("main done")


def run_scenarios(n_seeds=20):
    rows = []
    for sc in list(SCENARIOS):
        for s in range(n_seeds):
            raw = generate(N_ENROLLED, sc, 5000 + s)
            d = raw[(~raw.invalid_hba1c) & (~raw.no_followup)].reset_index(drop=True)
            tr, te = train_test_split(np.arange(len(d)), test_size=0.2, stratify=d.response, random_state=s)
            aucs = {}
            for fs in ["clinical", "clinical+natto"]:
                gs = fit_tuned("xgboost", d.iloc[tr][FEATURE_SETS[fs]], d.response.values[tr], small=True, seed=s)
                aucs[fs] = roc_auc_score(d.response.values[te], gs.best_estimator_.predict_proba(d.iloc[te][FEATURE_SETS[fs]])[:, 1])
            c = causal(d, B=0 + 1)
            rows.append(dict(scenario=sc, seed=s, dauc=aucs["clinical+natto"] - aucs["clinical"],
                             auc_clin=aucs["clinical"], adj_delta=c["adj_delta"]["est"],
                             ipw_delta=c["ipw_delta"]["est"], ipw_or=c["ipw_or"]["est"],
                             crude_or=c["crude_or"]["est"]))
            print(sc, s, flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "scenario_runs.csv", index=False)
    agg = {}
    for sc, g in df.groupby("scenario"):
        agg[sc] = dict(true_b=SCENARIOS[sc]["b_high"], n=len(g),
                       dauc_mean=float(g.dauc.mean()), dauc_sd=float(g.dauc.std()),
                       dauc_pos=float((g.dauc > 0).mean()),
                       adj_mean=float(g.adj_delta.mean()), adj_sd=float(g.adj_delta.std()),
                       ipw_mean=float(g.ipw_delta.mean()), ipw_sd=float(g.ipw_delta.std()),
                       ipw_or_mean=float(g.ipw_or.mean()), crude_or_mean=float(g.crude_or.mean()))
    json.dump(agg, open(OUT / "scenario_summary.json", "w"), indent=1)
    print("scenarios done")


if __name__ == "__main__":
    {"main": run_main, "scenarios": run_scenarios}[sys.argv[1]]()
