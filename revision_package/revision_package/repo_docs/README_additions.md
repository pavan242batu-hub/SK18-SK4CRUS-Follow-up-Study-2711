## Revision 1 additions (follow-up to reviewer comments)

### What this repository now also contains
- `scripts/12_alt_predictors_lr_xgb.py`, `scripts/14_*` : alternative predictors (logistic regression, XGBoost, calibrated linear SVM) and paired selector (AURC) comparison.
- `scripts/16_retrieval_stability_saved_outputs.py` : label agreement, evidence consistency and depth-prefix stability, computed from saved outputs.
- `scripts/17_retrieval_sensitivity_rerun.py` : one-factor-at-a-time retrieval-configuration sensitivity rerun on the development partition.
- `results/` : summary tables used in the manuscript (see `results/README.md`).

### Scope and limits (please read)
- The historical test partition was previously reported and is **not** an untouched holdout. Nothing in this revision was tuned on it.
- No independent external dataset was evaluated; no claim of external generalization is made.
- Adding retrieval features did not produce a distinguishable AURC gain for any of the three alternative predictors (all 95% intervals include zero).
- No formal risk-control guarantee is claimed; calibration results are empirical.
- A transformer predictor was not evaluated (no GPU available in the run environment).
- Research evaluation only; not a clinical decision tool.

### Reproducing
1. Environment: Python with `numpy pandas scikit-learn torch scipy rank_bm25 sentence-transformers` (versions: see `requirements.txt` / environment record).
2. Scripts 16 and 17 were run in Google Colab with the project on Google Drive; set the `workspace`/`ws` path at the top of each script.
3. Inputs are verified by SHA-256 before use (see `protocol.json` in each output folder). The base configuration is checked to reproduce saved validation output before any variant runs; the script stops otherwise.
4. Seeds: model seed 42 (from the plan), analysis seed 20261121. Bootstrap: group-level resampling by `group_id` (2000 replicates in script 16, 1000 in script 17).
5. Each analysis writes a new timestamped folder and never overwrites earlier outputs.

### Data
The dialogue text is not redistributed here. Obtain MTS-Dialog from its original source and follow its license. Record-level outputs contain record IDs and numeric scores only.
