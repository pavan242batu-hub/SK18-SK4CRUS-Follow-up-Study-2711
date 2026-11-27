# Files to copy from Google Drive into the repo (not included in this zip)

Drive root: /content/drive/MyDrive/SK4_recovery_20261121/project/extensions_ac/

Scripts (wherever you saved them; I did not have their code):
- 12_alt_predictors_lr_xgb.py  -> scripts/
- 14 (alt selector analysis script) -> scripts/
- Any ablation / calibration / class-wise scripts used for comments 6-8 -> scripts/

Output folders - copy ONLY small summary files (CSV summaries, protocol.json, run report):
- alt_predictors_v1_20261126T141853693631Z
- alt_selector_analysis_v1_20261126T142821427072Z
- retrieval_stability_v1_20261126T144914301572Z   (summary CSVs + protocol.json; record-level CSV optional)
- retrieval_sensitivity_v1_20261126T145650760105Z (summary_*.csv + protocol.json; records_*.csv optional)
Replace the *_TRANSCRIBED.csv files in results/ with the originals.

Do NOT copy: reference_embeddings.npz, any file containing dialogue text, model caches, tokens, .ipynb with outputs showing text.
Check: records_*.csv hold record IDs, labels, predictions, scores and document IDs only (no text). Confirm the dataset license allows this.
