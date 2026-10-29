"""Synthetic "JDCS-like" cohort generator (Section 2.7 of the manuscript).

Every effect parameter is declared in SCENARIOS and can be changed. The records
are simulated; they are not JDCS participants and carry no clinical meaning.
"""
import numpy as np
import pandas as pd

# b_high: true effect of >=4 servings/week on change in HbA1c (NGSP points),
#         conditional on all covariates. The moderate category gets half of it.
# dq_out: effect of the latent diet-quality score on change in HbA1c (per SD).
# dq_nat: effect of diet quality on the latent natto propensity (per SD).
# dq_obs: noise SD of the observed proxies of diet quality (higher = worse proxy).
SCENARIOS = {
    "null":       dict(b_high=0.00, dq_out=-0.10, dq_nat=0.35, dq_obs=1.0),
    "weak":       dict(b_high=-0.10, dq_out=-0.10, dq_nat=0.35, dq_obs=1.0),
    "moderate":   dict(b_high=-0.30, dq_out=-0.10, dq_nat=0.35, dq_obs=1.0),
    "confounded": dict(b_high=0.00, dq_out=-0.30, dq_nat=0.80, dq_obs=2.5),
}

CLINICAL = ["age", "male", "bmi", "duration", "hba1c0", "fpg", "tg", "hdl", "egfr",
            "hb", "insulin", "metformin", "sulfonylurea", "dpp4", "vka", "smoking",
            "arm", "region_west"]
LIFESTYLE = ["activity", "energy"]
DIET = ["rice", "other_soy", "veg", "fish"]
NATTO = ["natto_cat"]
FEATURE_SETS = {
    "clinical": CLINICAL,
    "diet": DIET + ["energy"],
    "clinical+natto": CLINICAL + NATTO,
    "full": CLINICAL + LIFESTYLE + DIET + NATTO,
}
CATS = {0: "never", 1: "<1/week", 2: "1-3/week", 3: ">=4/week"}


def _sig(x):
    return 1 / (1 + np.exp(-x))


def generate(n=9000, scenario="moderate", seed=1, shifted=False):
    p = SCENARIOS[scenario]
    r = np.random.default_rng(seed)
    west = r.random(n) < (0.30 if shifted else 0.45)
    male = r.random(n) < 0.55
    age = np.clip(r.normal(66 if shifted else 59, 9, n), 30, 85)
    dq = r.normal(0, 1, n) + 0.3 * (age - 59) / 9 + 0.2 * (~male)
    bmi = np.clip(r.normal(23.0, 3.2, n) + (1.5 if shifted else 0), 15, 42)
    duration = np.clip(r.gamma(3.0, 3.6, n) + 0.05 * (age - 59), 0.2, 45)
    hba1c0 = np.clip(r.normal(8.6 if shifted else 8.0, 1.2, n) + 0.015 * duration, 6.0, 13.5)
    fpg = np.clip(28.7 * hba1c0 - 46.7 + r.normal(0, 28, n), 70, 400)
    tg = np.exp(r.normal(4.85 + 0.03 * (bmi - 23), 0.45, n))
    hdl = np.clip(r.normal(56 - 1.0 * (bmi - 23) - 6 * male, 13, n), 20, 120)
    egfr = np.clip(r.normal(82 - 0.55 * (age - 59) - 0.35 * duration, 15, n), 12, 130)
    hb = np.clip(r.normal(14.0 - 1.4 * (~male) + 0.01 * (egfr - 80), 1.2, n), 7, 19)
    insulin = r.random(n) < _sig(-2.0 + 0.55 * (hba1c0 - 8) + 0.06 * (duration - 10))
    metformin = r.random(n) < 0.55
    sulfonylurea = r.random(n) < _sig(-0.8 - 0.4 * insulin)
    dpp4 = r.random(n) < 0.22
    vka = r.random(n) < _sig(-4.2 + 0.05 * (age - 59))
    smoking = np.where(r.random(n) < 0.18 + 0.15 * male, 2, np.where(r.random(n) < 0.35, 1, 0))
    arm = r.random(n) < 0.5
    activity = np.exp(r.normal(3.0 + 0.08 * dq, 0.7, n))
    energy = np.clip(r.normal(1750 + 280 * male, 330, n), 900, 3500)
    rice = np.clip(r.normal(2.6 - 0.15 * dq, 0.7, n), 0, 6)
    veg = np.clip(3.5 + 0.8 * dq + r.normal(0, p["dq_obs"], n), 0, 12)
    fish = np.clip(3.0 + 0.7 * dq + r.normal(0, 1.5 * p["dq_obs"], n), 0, 14)
    other_soy = np.clip(r.normal(5.0 + 0.6 * dq + 0.2 * (~west), 2.0, n), 0, 14)

    az = (activity - activity.mean()) / activity.std()
    u = (-0.6 * west + p["dq_nat"] * dq + 0.02 * (age - 59) + 0.25 * az
         - 2.5 * vka + 0.1 * (~male) + r.logistic(0, 1, n))
    cuts = np.quantile(u, [0.22, 0.40, 0.70])
    cat = np.searchsorted(cuts, u)

    eff = np.array([0.0, 0.0, p["b_high"] / 2, p["b_high"]])[cat]
    delta = (-0.30 * (hba1c0 - 7.9) - 0.20 * (hba1c0 > 9.0) - 0.15 * arm
             + 0.15 * (insulin & (duration > 15)) - 0.08 * metformin
             + p["dq_out"] * dq - 0.002 * (age - 59)
             + eff + r.normal(0, 0.85, n))

    df = pd.DataFrame(dict(
        age=age, male=male.astype(int), bmi=bmi, duration=duration, hba1c0=hba1c0,
        fpg=fpg, tg=tg, hdl=hdl, egfr=egfr, hb=hb, insulin=insulin.astype(int),
        metformin=metformin.astype(int), sulfonylurea=sulfonylurea.astype(int),
        dpp4=dpp4.astype(int), vka=vka.astype(int), smoking=smoking,
        arm=arm.astype(int), region_west=west.astype(int), activity=activity,
        energy=energy, rice=rice, other_soy=other_soy, veg=veg, fish=fish,
        natto_cat=cat.astype(float), delta=delta, hba1c12=hba1c0 + delta))
    df["response"] = (df["delta"] <= -0.5).astype(int)
    df["tg_hdl"] = df["tg"] / df["hdl"]

    # exclusions: invalid HbA1c conditions, no follow-up inside the window
    df["invalid_hba1c"] = r.random(n) < 0.025
    df["no_followup"] = r.random(n) < (0.06 + 0.03 * insulin)
    # missingness in baseline measures; natto missing is kept distinct from zero
    for col, pm in [("fpg", 0.04), ("tg", 0.05), ("hdl", 0.05), ("egfr", 0.03),
                    ("hb", 0.06), ("activity", 0.08), ("natto_cat", 0.03)]:
        df.loc[r.random(n) < pm, col] = np.nan
    return df
