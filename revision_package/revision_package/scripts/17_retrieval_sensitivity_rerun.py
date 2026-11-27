# Script 17 - retrieval-configuration sensitivity RERUN on the development partition (reviewer comments 2 and 5).
# Rebuilds CascadeCore exactly as run_validation.py does, verifies the base config reproduces saved validation output,
# then changes ONE setting at a time. No setting is selected or tuned. Writes a NEW timestamped folder.
# Requires: pip install rank_bm25 sentence-transformers ; Colab with Drive mounted (CPU runtime).
# Code body is identical to the cell executed on 2026-11-26/27 (output: retrieval_sensitivity_v1_20261126T145650760105Z).

from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math, sys, time, importlib.util
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score

ws = Path("/content/drive/MyDrive/SK4_recovery_20261121")
ext = ws / "project/extensions_ac"; tools = ws / "project/tools/followup_v07"
assert ext.is_dir(), "Mount Drive first"
SEED, REPS = 20261121, 1000
RESUME_DIR = None
RUN_ONLY = None

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
EXPECT = {
 tools/"scripts/cascade_core.py": "2a7585820ca048e43b57e48f31c8a88901a0524908b2556b99c5bd805d1db691",
 tools/"provenance/recovery_plan.json": "90c5a0f15bcf5cd91910949d05705205fa7dc680a0d58f7c61592e784958d0cd",
 tools/"reference/historical_split_manifest.csv": "ca62ec75e905a4dbf4508ad172fe33e50f7b7779e4a90ad6f58c59c91a7c4d91",
 ws/"project/data/pilot_v1/research_data.csv": "55fe0e91f01e10742058fdfc07cd416e5b052c3cddf6c166bd7b341b50ddeed3"}
for p, h in EXPECT.items(): assert sha(p) == h, f"hash mismatch: {p.name}"

sys.path.insert(0, str(tools / "scripts"))
spec = importlib.util.spec_from_file_location("run_validation_mod", tools/"scripts/run_validation.py")
rvm = importlib.util.module_from_spec(spec); spec.loader.exec_module(rvm)
from cascade_core import CascadeCore
from sentence_transformers import SentenceTransformer, CrossEncoder

plan = json.loads((tools/"provenance/recovery_plan.json").read_text())
cfg0 = plan["method"]["retrieval_configuration"]; unc = plan["method"]["uncertainty_configuration"]
data = pd.read_csv(ws/"project/data/pilot_v1/research_data.csv", dtype=str, keep_default_na=False)
man = pd.read_csv(tools/"reference/historical_split_manifest.csv", dtype=str, keep_default_na=False)
aligned = data.set_index("record_id", verify_integrity=True).loc[man.record_id.tolist()]
assert np.array_equal(aligned["label"].to_numpy(), man["label"].to_numpy())
man["text"] = aligned["text"].map(rvm.normalize).to_numpy()
assert man.text.map(lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()).eq(man.dialogue_hash).all()
train = man[man.split.eq("train")].copy(); queries = man[man.split.eq("validation")].copy()
classes = sorted(man.label.unique().tolist()); assert classes == plan["reference"]["classes"]
ident = json.dumps(train[["record_id","label","dialogue_hash"]].to_dict(orient="records"),
                   ensure_ascii=False, separators=(",", ":")).encode("utf-8")
assert hashlib.sha256(ident).hexdigest() == plan["reference"]["ordered_identity_sha256"]
print("train", len(train), "dev queries", len(queries))

rt = plan["runtime"]; mods = plan["models"]; torch.manual_seed(cfg0["seed"])
def load(cls, mid, rev, **kw):
    try: return cls(mid, revision=rev, local_files_only=True, trust_remote_code=False, **kw)
    except Exception:
        print("not cached; downloading pinned revision", mid)
        return cls(mid, revision=rev, trust_remote_code=False, **kw)
dense = load(SentenceTransformer, mods["dense"]["model_id"], mods["dense"]["resolved_revision"], device=rt["device"])
dense.max_seq_length = rt["dense_max_sequence_length"]; dense.eval()
ce = load(CrossEncoder, mods["cross_encoder"]["model_id"], mods["cross_encoder"]["resolved_revision"],
          device=rt["device"], max_length=rt["cross_encoder_max_sequence_length"])
ce.model.eval()

npz_path = list((ws/"project/extensions_v07").glob("recovery_validation_*/reference_embeddings.npz"))
assert len(npz_path) == 1; z = np.load(npz_path[0], allow_pickle=True)
assert np.array_equal(z["record_ids"].astype(str), train.record_id.to_numpy()), "embedding order mismatch"
embs = z["embeddings"]
core = CascadeCore(texts=train.text.tolist(), labels=train.label.tolist(), record_ids=train.record_id.tolist(),
                   embeddings=embs, classes=classes, embedder=dense, cross_encoder=ce,
                   retrieval_config=cfg0, uncertainty_config=unc)
_orig = core.ce_raw_logits; ce_cache = {}
def cached(pairs):
    pairs = [tuple(p) for p in pairs]
    miss = [p for p in dict.fromkeys(pairs) if p not in ce_cache]
    if miss:
        for p, v in zip(miss, _orig(miss)): ce_cache[p] = float(v)
    return np.array([ce_cache[p] for p in pairs], float)
core.ce_raw_logits = cached
qemb_cache = {}
def qemb(q):
    if q not in qemb_cache:
        qemb_cache[q] = dense.encode([q], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
    return qemb_cache[q]

def perturb(text, kind, rid):
    w = text.split()
    if kind is None: return text
    if kind == "trunc80": return " ".join(w[:max(1, math.ceil(0.8 * len(w)))])
    p = {"drop10": 0.10, "drop20": 0.20}[kind]
    s = int(hashlib.sha256(f"{SEED}|{rid}|{kind}".encode()).hexdigest()[:8], 16)
    keep = np.random.default_rng(s).random(len(w)) >= p
    if not keep.any(): keep[0] = True
    return " ".join(np.asarray(w)[keep])

CONFIGS = {
 "base": ({}, "Cascade", None),
 "rrf_top_n_5": ({"rrf_top_n": 5}, "Cascade", None), "rrf_top_n_15": ({"rrf_top_n": 15}, "Cascade", None),
 "topk_retrieve_10": ({"topk_retrieve": 10}, "Cascade", None), "topk_retrieve_40": ({"topk_retrieve": 40}, "Cascade", None),
 "shortlist_3": ({"cascade_top_k_classes": 3}, "Cascade", None), "shortlist_7": ({"cascade_top_k_classes": 7}, "Cascade", None),
 "no_shortlist_filter": ({}, "SingleStageTruncated", None),
 "query_drop10": ({}, "Cascade", "drop10"), "query_drop20": ({}, "Cascade", "drop20"),
 "query_truncate80": ({}, "Cascade", "trunc80")}

def run_config(name):
    override, method, kind = CONFIGS[name]
    core.RETRIEVAL_CFG = {**cfg0, **override}; rows = []; t0 = time.time()
    for n, r in enumerate(queries.itertuples(index=False), 1):
        q = perturb(r.text, kind, r.record_id)
        with torch.inference_mode():
            cand = core.prepare_candidates(q); res = core.score_documents(q, cand, method)
        comp = np.asarray(res["components"], float); idx = [i for i, _ in res["documents"]]
        sims = embs[idx] @ qemb(q)
        rows.append(dict(record_id=r.record_id, group_id=r.group_id, true_label=r.label, prediction=res["pred"],
            correct=int(res["pred"] == r.label), confidence=res["confidence"], H_pred=comp[0], H_retrieval=comp[1],
            V_RRF=comp[2], CRUS=core.composite_crus(comp), fallback=int(res["fallback"]), n_docs=len(idx),
            sem_sim_mean=float(sims.mean()), sem_sim_max=float(sims.max()),
            doc_ids="|".join(train.record_id.to_numpy()[idx]), shortlist="|".join(res["shortlist"])))
        if n % 50 == 0: print(name, n, f"{time.time()-t0:.0f}s")
    return pd.DataFrame(rows)

if RESUME_DIR: out_dir = ext / RESUME_DIR; assert out_dir.is_dir()
else:
    out_dir = ext / ("retrieval_sensitivity_v1_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out_dir.mkdir(exist_ok=False)
def path_of(name): return out_dir / f"records_{name}.csv"

base_path = path_of("base")
if not base_path.exists():
    base_df = run_config("base")
    ref_p = list((ws/"project/extensions_v07").glob("recovery_validation_*/fresh_validation_predictions.jsonl"))
    assert len(ref_p) == 1
    ref = pd.DataFrame([json.loads(l) for l in ref_p[0].open(encoding="utf-8")]).set_index("record_id").loc[base_df.record_id]
    ok_pred = (ref["prediction"].to_numpy() == base_df.prediction.to_numpy()).all()
    dif = {k: float(np.abs(ref[k].to_numpy(float) - base_df[k].to_numpy(float)).max())
           for k in ["confidence", "H_pred", "H_retrieval", "V_RRF", "CRUS"]}
    print("base reproduces saved predictions:", ok_pred, dif)
    if not ok_pred or max(dif.values()) > 1e-4:
        raise RuntimeError("Base configuration does not reproduce saved validation output; nothing else was run.")
    base_df.to_csv(base_path, index=False)
base = pd.read_csv(base_path, dtype={"record_id": str, "group_id": str})
for name in CONFIGS:
    if name == "base" or (RUN_ONLY and name not in RUN_ONLY) or path_of(name).exists(): continue
    run_config(name).to_csv(path_of(name), index=False); print("saved", name)

codes, uniq = pd.factorize(base.group_id); G = len(uniq); rng = np.random.default_rng(SEED)
W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G)[codes] for _ in range(REPS)])
def metrics(df, w=None):
    e = 1 - df.correct.to_numpy(); w = np.ones(len(df)) if w is None else w
    out = {"accuracy": float((w * df.correct).sum() / w.sum())}
    k = w > 0
    for nm, col in [("AUROC_CRUS", "CRUS"), ("AUROC_H_retrieval", "H_retrieval")]:
        out[nm] = np.nan if (w * e).sum() in (0, w.sum()) else roc_auc_score(e[k], df[col].to_numpy()[k], sample_weight=w[k])
    return out
def ci(a): a = np.asarray(a, float); a = a[~np.isnan(a)]; return np.percentile(a, 2.5), np.percentile(a, 97.5)
base_pt = metrics(base); base_bs = [metrics(base, w) for w in W]
summ = []
for name in CONFIGS:
    if not path_of(name).exists(): continue
    df = pd.read_csv(path_of(name), dtype={"record_id": str, "group_id": str})
    assert (df.record_id == base.record_id).all()
    pt = metrics(df); bs = [metrics(df, w) for w in W]
    row = {"config": name, "n": len(df)}
    for m in ["accuracy", "AUROC_CRUS", "AUROC_H_retrieval"]:
        lo, hi = ci([b[m] for b in bs]); row[m] = pt[m]; row[m + "_lo"] = lo; row[m + "_hi"] = hi
        if name != "base":
            dlo, dhi = ci([b[m] - c[m] for b, c in zip(bs, base_bs)])
            row[m + "_diff_vs_base"] = pt[m] - base_pt[m]; row[m + "_diff_lo"] = dlo; row[m + "_diff_hi"] = dhi
    sa = [set(a.split("|")) for a in base.doc_ids]; sb = [set(a.split("|")) for a in df.doc_ids]
    row["pred_agreement_with_base"] = float((df.prediction == base.prediction).mean())
    row["doc_set_jaccard_mean"] = float(np.mean([len(a & b) / len(a | b) for a, b in zip(sa, sb)]))
    row["top1_doc_same"] = float(np.mean([a.split("|")[0] == b.split("|")[0] for a, b in zip(base.doc_ids, df.doc_ids)]))
    row["mean_abs_dH_retrieval"] = float((df.H_retrieval - base.H_retrieval).abs().mean())
    row["mean_abs_dCRUS"] = float((df.CRUS - base.CRUS).abs().mean())
    row["sem_sim_mean"] = float(df.sem_sim_mean.mean()); row["fallback_rate"] = float(df.fallback.mean()); row["mean_n_docs"] = float(df.n_docs.mean())
    summ.append(row)
S = pd.DataFrame(summ); sp = out_dir / f"summary_{datetime.now(timezone.utc).strftime('%H%M%S')}.csv"
S.to_csv(sp, index=False)
(out_dir / "protocol.json").write_text(json.dumps(dict(
    partition="development (validation) only", seed=SEED, bootstrap_reps=REPS, base_config=cfg0,
    configs={k: dict(override=v[0], method=v[1], perturbation=v[2]) for k, v in CONFIGS.items()},
    input_sha256={str(k.relative_to(ws)): v for k, v in EXPECT.items()},
    notes=["One factor changed at a time against base; no setting was selected or tuned.",
           "no_shortlist_filter uses the SingleStageTruncated method (rrf_top_n documents, no shortlist filter).",
           "Query perturbation affects the query only; reference documents are unchanged."]), indent=1))
cols = ["config", "accuracy", "AUROC_CRUS", "AUROC_CRUS_diff_vs_base", "AUROC_CRUS_diff_lo", "AUROC_CRUS_diff_hi",
        "pred_agreement_with_base", "doc_set_jaccard_mean", "mean_abs_dH_retrieval", "sem_sim_mean"]
print(S[[c for c in cols if c in S.columns]].round(4).to_string(index=False)); print("Saved to:", out_dir.name)
