# natto-hba1c-pipeline

Leakage-resistant machine-learning pipeline for predicting HbA1c response from natto consumption frequency in type 2 diabetes, verified on synthetic JDCS-like records.

All records are synthetic and generated with declared parameters. Nothing here supports a clinical claim about natto. The pipeline is specified for a later analysis of the Japan Diabetes Complications Study, whose records were not available.

## Layout
- `code/natto_synth.py` synthetic cohort generator
- `code/natto_run.py` main pipeline (models, calibration, decision curves, propensity weighting, scenarios)
- `code/natto_features.py` feature-group analysis
- `code/natto_figs.py`, `code/natto_figs34.py` figures
- `results/` result tables, JSON summaries and figures

## Run
Python 3 with numpy, pandas, scikit-learn, statsmodels, xgboost, lightgbm, shap, matplotlib.

```
cd code
python natto_run.py main        # main analysis
python natto_run.py scenarios   # simulated scenarios
```

To use real data, replace the synthetic generator call in `natto_run.py`. Released under the MIT License (see `LICENSE`).
