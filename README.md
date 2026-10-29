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
Python 3 with numpy, pandas, scikit-learn, xgboost, lightgbm, shap, matplotlib. Run `python code/natto_run.py` from the repository root. To use real data, replace the loader in `natto_run.py`.
