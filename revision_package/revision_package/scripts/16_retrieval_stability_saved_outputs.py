# Script 16 - retrieval stability from SAVED outputs (reviewer comment 2).
# Uses saved follow-up outputs only; loads no models; writes a NEW timestamped folder (never overwrites).
# Run in Google Colab with Drive mounted. Set `workspace` below if your location differs.
# Code body is identical to the cell executed on 2026-11-26 (output: retrieval_stability_v1_20261126T144914301572Z).

from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

workspace = Path("/content/drive/MyDrive/SK4_recovery_20261121")
ext = workspace / "project/extensions_ac"
assert ext.is_dir(), "Mount Drive first"
REPS, SEED, DEPTHS = 2000, 20261121, (3, 5, 7)

def only(prefix):
    m = [p for p in ext.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if len(m) != 1: raise RuntimeError(f"Expected one folder for {prefix}; found {[p.name for p in m]}")
    return m[0]
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

run = only("historical_test_followup_")
report = json.loads((run / "followup_run_report.json").read_text())
audit_p, jl_p = run / "followup_record_audit.csv", run / "fresh_followup_predictions.jsonl"
assert sha(audit_p) == report["output_sha256"]["followup_record_audit.csv"], "audit hash mismatch"
h_jl = report.get("output_sha256", {}).get("fresh_followup_predictions.jsonl")
if h_jl: assert sha(jl_p) == h_jl, "predictions hash mismatch"
audit = pd.read_csv(audit_p, dtype={"record_id": str, "group_id": str, "true_label": str, "prediction": str})

recs = {}
with jl_p.open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        recs[str(r["record_id"])] = dict(ids=[str(x) for x in r["retrieved_training_ids"]],
            shortlist=[str(x) for x in r["shortlist"]], H_ret=float(r["H_retrieval"]), fallback=bool(r["fallback"]))
man = pd.read_csv(workspace / "project/tools/followup_v07/reference/historical_split_manifest.csv",
                  usecols=["record_id", "label", "split"], dtype=str)
assert man.record_id.is_unique
lab = dict(zip(man.record_id, man.label)); spl = dict(zip(man.record_id, man.split))
classes = sorted(set(man.label)); NC = len(classes)
assert NC == 17, f"expected 17 classes, found {NC}"
missing = [i for r in recs.values() for i in r["ids"] if i not in lab]
assert not missing, f"{len(missing)} retrieved IDs not in manifest"
assert set(audit.record_id) <= set(recs), "audit records missing from predictions"
print("splits of retrieved IDs:", Counter(spl[i] for r in recs.values() for i in r["ids"]))

def ent(labels):
    c = Counter(labels); p = np.array([c.get(k, 0) for k in classes], float) / len(labels)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(NC))
def maj(labels):
    c = Counter(labels); return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

rows = []
for _, a in audit.iterrows():
    r = recs[a.record_id]; L = [lab[i] for i in r["ids"]]
    row = dict(record_id=a.record_id, group_id=a.group_id, error=int(a.prediction_error), n_docs=len(L),
               fallback=int(r["fallback"]), H_saved=r["H_ret"], H_recomputed=ent(L),
               agree_pred=float(np.mean([x == a.prediction for x in L])),
               agree_true=float(np.mean([x == a.true_label for x in L])),
               top1_is_pred=int(L[0] == a.prediction), majority_share=Counter(L).most_common(1)[0][1] / len(L),
               majority_is_pred=int(maj(L) == a.prediction), n_distinct_labels=len(set(L)),
               true_in_shortlist=int(a.true_label in r["shortlist"]), H_k10=ent(L))
    for k in DEPTHS:
        row[f"H_k{k}"] = ent(L[:k]); row[f"majority_k{k}_same_as_k10"] = int(maj(L[:k]) == maj(L))
    rows.append(row)
rec = pd.DataFrame(rows)
maxdiff = float((rec.H_saved - rec.H_recomputed).abs().max())
print("max |saved H_retrieval - recomputed|:", maxdiff)
if maxdiff > 1e-6:
    raise RuntimeError("Saved IDs do not reproduce H_retrieval; stop and send this message.")

score_cols = {"1-agree_with_prediction": 1 - rec.agree_pred, "1-top1_is_prediction": 1 - rec.top1_is_pred,
              "1-majority_share": 1 - rec.majority_share, "n_distinct_labels": rec.n_distinct_labels}
for k in DEPTHS: score_cols[f"H_retrieval_depth{k}_prefix"] = rec[f"H_k{k}"]
score_cols["H_retrieval_depth10"] = rec.H_k10
ref = audit.set_index("record_id").loc[rec.record_id]
for c in ["predictive_only_error_ranking_score", "native_predictive_plus_retrieval_error_ranking_score",
          "fixed_CRUS_error_ranking_score", "predictive_entropy_error_ranking_score"]:
    score_cols["REF:" + c] = ref[c].to_numpy()

err = rec.error.to_numpy(); codes, uniq = pd.factorize(rec.group_id); G = len(uniq)
rng = np.random.default_rng(SEED)
W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G)[codes] for _ in range(REPS)])
def auc(s, w=None):
    if w is None: return roc_auc_score(err, s)
    k = w > 0
    if (w * err).sum() in (0, w.sum()): return np.nan
    return roc_auc_score(err[k], s[k], sample_weight=w[k])
out = []
for name, s in score_cols.items():
    s = np.asarray(s, float); b = np.array([auc(s, w) for w in W]); b = b[~np.isnan(b)]
    out.append(dict(score=name, error_AUROC=auc(s), ci95_lo=np.percentile(b, 2.5), ci95_hi=np.percentile(b, 97.5), valid_reps=len(b)))
auroc = pd.DataFrame(out)

bins = pd.cut(rec.agree_pred, [-0.001, 0.4999, 0.7999, 0.9999, 1.0], labels=["<0.5", "0.5-0.79", "0.8-0.99", "1.0"])
bt = rec.groupby(bins, observed=True).agg(n=("error", "size"), errors=("error", "sum")).reset_index()
bt["error_rate"] = bt.errors / bt.n
stab = pd.DataFrame([dict(depth=k, mean_abs_H_diff_vs_10=float((rec[f"H_k{k}"] - rec.H_k10).abs().mean()),
        share_majority_label_unchanged=float(rec[f"majority_k{k}_same_as_k10"].mean())) for k in DEPTHS])

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
out_dir = ext / f"retrieval_stability_v1_{stamp}"; out_dir.mkdir(exist_ok=False)
rec.to_csv(out_dir / "record_level_retrieval_measures.csv", index=False)
auroc.to_csv(out_dir / "error_detection_auroc_group_bootstrap.csv", index=False)
bt.to_csv(out_dir / "error_rate_by_label_agreement.csv", index=False)
stab.to_csv(out_dir / "depth_prefix_stability.csv", index=False)
(out_dir / "protocol.json").write_text(json.dumps(dict(
    inputs_sha256=dict(audit=sha(audit_p), predictions=sha(jl_p)), max_abs_H_diff=maxdiff, reps=REPS, seed=SEED,
    notes=["Saved outputs only; no models loaded; no thresholds or selectors tuned.",
           "Depth analysis is prefix truncation of the saved final document set, not a rerun at a smaller rrf_top_n.",
           "agree_true and true_in_shortlist use the true label: diagnostic only, not available at deployment.",
           "Semantic similarity is not computed here; it needs the dense encoder."]), indent=1))
print(auroc.round(4).to_string(index=False)); print(bt.to_string(index=False)); print(stab.round(4).to_string(index=False))
print("Saved to:", out_dir.name)
