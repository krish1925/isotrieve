import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, ir_datasets
from sentence_transformers import SentenceTransformer
SEED = 20260829
ds = ir_datasets.load("beir/scifact")
texts = [f"{d.title} {d.text}".strip() for d in ds.docs_iter()]
rng = np.random.default_rng(SEED)
idx = rng.permutation(len(texts))
cal_texts = [texts[i] for i in idx[:2000]]
hold_texts = [texts[i] for i in idx[2000:2200]]
src = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
tgt = SentenceTransformer("BAAI/bge-large-en-v1.5")
E_src = src.encode(cal_texts, normalize_embeddings=True, show_progress_bar=False)
E_tgt = tgt.encode(cal_texts, normalize_embeddings=True, show_progress_bar=False)
h_src = src.encode(hold_texts, normalize_embeddings=True, show_progress_bar=False)
h_tgt = tgt.encode(hold_texts, normalize_embeddings=True, show_progress_bar=False)
tmp = Path(tempfile.mkdtemp(prefix="iso_p7real_"))
np.save(tmp/"src.npy", E_src); np.save(tmp/"tgt.npy", E_tgt)
np.save(tmp/"hs.npy", h_src); np.save(tmp/"ht.npy", h_tgt)
cli = str(Path(sys.executable).parent / "isotrieve")
r_cal = subprocess.run([cli,"calibrate","--source-vectors",str(tmp/"src.npy"),"--target-vectors",str(tmp/"tgt.npy"),"--output",str(tmp/"m.isotrieve")],capture_output=True,text=True,timeout=600)
r_pass = subprocess.run([cli,"gate","--mapping",str(tmp/"m.isotrieve"),"--source-vectors",str(tmp/"hs.npy"),"--target-vectors",str(tmp/"ht.npy"),"--format","json"],capture_output=True,text=True,timeout=600)
try:
    d = json.loads(r_pass.stdout)
    print("P7-04b-real:", r_pass.returncode, {k: d.get(k) for k in ("verdict","predicted_retention","top1_retention","margin_compression")})
except Exception:
    print("P7-04b-real: rc=", r_pass.returncode, r_pass.stdout[-300:], r_pass.stderr[-300:])
# degenerate arm: permuted targets
p = np.random.default_rng(SEED+1).permutation(200)
r_fail = subprocess.run([cli,"gate","--mapping",str(tmp/"m.isotrieve"),"--source-vectors",str(tmp/"hs.npy"),"--target-vectors",str(tmp/"ht.npy"[0:0] or str(tmp/"ht.npy"))],capture_output=True,text=True,timeout=600) if False else None
np.save(tmp/"ht_bad.npy", h_tgt[p])
r_fail = subprocess.run([cli,"gate","--mapping",str(tmp/"m.isotrieve"),"--source-vectors",str(tmp/"hs.npy"),"--target-vectors",str(tmp/"ht_bad.npy"),"--format","json"],capture_output=True,text=True,timeout=600)
print("P7-04b-real-degenerate exit:", r_fail.returncode)
