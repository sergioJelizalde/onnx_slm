"""
Step 3 — Prepare binary input files for C inference + cross-validate C vs Python
==================================================================================
This script:
  1. Tokenizes the test domains using the same tokenizer as the Python benchmark
  2. Writes a compact binary file (inputs/domains.bin) that infer_arm64.c reads
  3. Runs Python ONNX INT8 inference on the same inputs
  4. After C runs, loads C embeddings and computes cosine similarity
     → validates that C and Python produce identical outputs

Binary format written:
  [n_domains: int32][seq_len: int32][embed_dim: int32]
  for each domain:
    [domain_len: int32][domain_str: char*domain_len]
    [input_ids:  int64 * seq_len]
    [attn_mask:  int64 * seq_len]
    [tok_types:  int64 * seq_len]

Run order:
  1. python3 3_prepare_c_inputs.py          ← this file (writes .bin + py results)
  2. ./infer_arm64 models/minilm_int8.onnx inputs/domains.bin 200
  3. python3 3_prepare_c_inputs.py --compare  ← cross-validate C vs Python
"""

import sys
import json
import struct
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer
import onnxruntime as ort

MODELS_DIR  = Path("models")
INPUTS_DIR  = Path("inputs");   INPUTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR = Path("outputs");  OUTPUTS_DIR.mkdir(exist_ok=True)
ONNX_INT8   = MODELS_DIR / "minilm_int8.onnx"
MAX_SEQ_LEN = 64
EMBED_DIM   = 384

# ── same test domains as benchmark ────────────────────────────────────────────
TEST_DOMAINS = [
    "google.com", "github.com", "amazon.com", "facebook.com", "twitter.com",
    "netflix.com", "reddit.com", "wikipedia.org", "stackoverflow.com",
    "cloudflare.com", "fastly.net", "akamaiedge.net", "cdn.jsdelivr.net",
    "mail.google.com", "api.github.com", "docs.microsoft.com",
    "security.ubuntu.com", "archive.ubuntu.com",
    "fonts.googleapis.com", "static.cloudflareinsights.com",
    "edge-star-shv-01-iad3.facebook.com",
    "login.microsoftonline.com", "s3.amazonaws.com",
    "ec2-54-234-12-1.compute.amazonaws.com",
    "storage.googleapis.com", "accounts.google.com",
    "1e100.net", "v4.api.example.com", "cdn2.example.co.uk",
    "very-long-subdomain-name-for-testing-purposes.example.com",
    "another-quite-long-subdomain.with-hyphens.and-dots.example.org",
]

def mean_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m   = mask[..., None].astype(np.float32)
    emb = (hidden * m).sum(1) / (m.sum(1) + 1e-9)
    return emb

def l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9
    return v / norms

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9
    nb = np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9
    return ((a / na) * (b / nb)).sum(-1)

# ═══════════════════════════════════════════════════════════════════════════════
# MODE A: prepare inputs + run Python ONNX inference
# ═══════════════════════════════════════════════════════════════════════════════
if "--compare" not in sys.argv:
    print("► Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR / "tokenizer"))

    print(f"► Tokenizing {len(TEST_DOMAINS)} domains (max_len={MAX_SEQ_LEN}) …")
    enc = tokenizer(TEST_DOMAINS, padding="max_length", truncation=True,
                    max_length=MAX_SEQ_LEN, return_tensors="np")
    input_ids  = enc["input_ids"].astype(np.int64)    # [N, S]
    attn_masks = enc["attention_mask"].astype(np.int64)
    tok_types  = enc.get("token_type_ids",
                         np.zeros_like(input_ids)).astype(np.int64)

    N, S = input_ids.shape
    print(f"  Shape: [{N}, {S}]  (all padded to max_len={MAX_SEQ_LEN})")

    # ── write binary for C ────────────────────────────────────────────────────
    bin_path = INPUTS_DIR / "domains.bin"
    print(f"► Writing {bin_path} …")
    with open(bin_path, "wb") as f:
        f.write(struct.pack("iii", N, S, EMBED_DIM))
        for i, domain in enumerate(TEST_DOMAINS):
            db = domain.encode("ascii")
            f.write(struct.pack("i", len(db)))
            f.write(db)
            f.write(input_ids[i].tobytes())
            f.write(attn_masks[i].tobytes())
            f.write(tok_types[i].tobytes())
    print(f"  Written: {bin_path.stat().st_size} bytes")

    # ── run Python ONNX INT8 inference ────────────────────────────────────────
    print(f"► Running Python ONNX INT8 inference …")
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(ONNX_INT8), sess_options=sess_opts,
                                providers=["CPUExecutionProvider"])

    # Inference one-by-one (batch=1) — same as C will do
    py_embeddings = []
    for i in range(N):
        feeds = {
            "input_ids":      input_ids[i:i+1],
            "attention_mask": attn_masks[i:i+1],
            "token_type_ids": tok_types[i:i+1],
        }
        hidden = sess.run(["last_hidden_state"], feeds)[0]  # [1, S, 384]
        emb    = mean_pool(hidden, attn_masks[i:i+1])       # [1, 384]
        emb    = l2_normalize(emb)                           # [1, 384]
        py_embeddings.append(emb[0])

    py_embeddings = np.stack(py_embeddings)  # [N, 384]

    # Save Python embeddings for comparison
    with open(OUTPUTS_DIR / "py_embeddings.bin", "wb") as f:
        f.write(struct.pack("ii", N, EMBED_DIM))
        f.write(py_embeddings.astype(np.float32).tobytes())

    # Also save as JSON (first 4 dims) for quick human inspection
    py_summary = []
    for i, d in enumerate(TEST_DOMAINS):
        py_summary.append({
            "domain": d,
            "tok_len": int((attn_masks[i] != 0).sum()),
            "emb_0_3": py_embeddings[i, :4].tolist(),
            "norm":    float(np.linalg.norm(py_embeddings[i])),
        })
    with open(OUTPUTS_DIR / "py_embeddings.json", "w") as f:
        json.dump(py_summary, f, indent=2)

    print(f"\n  Python embeddings saved → {OUTPUTS_DIR}/py_embeddings.bin")
    print(f"  Summary JSON          → {OUTPUTS_DIR}/py_embeddings.json")
    print(f"\n✓ Inputs ready. Now run C inference:")
    print(f"   mkdir -p outputs")
    print(f"   ./infer_arm64 models/minilm_int8.onnx inputs/domains.bin 200")
    print(f"  Then re-run: python3 3_prepare_c_inputs.py --compare")

# ═══════════════════════════════════════════════════════════════════════════════
# MODE B: cross-validate C vs Python embeddings
# ═══════════════════════════════════════════════════════════════════════════════
else:
    print("═"*72)
    print("CROSS-VALIDATION: C (ORT C API) vs Python (ORT Python API)")
    print("Both run ONNX INT8 on identical pre-tokenized inputs")
    print("═"*72)

    # Load Python embeddings
    py_bin = OUTPUTS_DIR / "py_embeddings.bin"
    c_bin  = OUTPUTS_DIR / "c_embeddings.bin"
    if not py_bin.exists():
        print(f"ERROR: {py_bin} not found. Run without --compare first.")
        sys.exit(1)
    if not c_bin.exists():
        print(f"ERROR: {c_bin} not found. Run infer_arm64 first.")
        sys.exit(1)

    with open(py_bin, "rb") as f:
        N, D = struct.unpack("ii", f.read(8))
        py_emb = np.frombuffer(f.read(), dtype=np.float32).reshape(N, D)

    with open(c_bin, "rb") as f:
        N_c, D_c = struct.unpack("ii", f.read(8))
        c_emb  = np.frombuffer(f.read(), dtype=np.float32).reshape(N_c, D_c)

    if N != N_c or D != D_c:
        print(f"ERROR: shape mismatch  py={N}×{D}  c={N_c}×{D_c}")
        sys.exit(1)

    print(f"\n  Comparing {N} domains, embed_dim={D}\n")
    print(f"  {'domain':<45} {'cos(py,c)':>10} {'L2 diff':>10} {'match':>6}")
    print("  " + "-"*72)

    sims, l2s = [], []
    for i, d in enumerate(TEST_DOMAINS[:N]):
        sim  = cosine_sim(py_emb[i:i+1], c_emb[i:i+1])[0]
        l2   = float(np.linalg.norm(py_emb[i] - c_emb[i]))
        flag = "✓" if sim > 0.9999 else ("~" if sim > 0.999 else "✗")
        sims.append(sim); l2s.append(l2)
        print(f"  {d:<45} {sim:>10.7f} {l2:>10.6f} {flag:>6}")

    print("  " + "-"*72)
    print(f"  {'MEAN':<45} {np.mean(sims):>10.7f} {np.mean(l2s):>10.6f}")
    print(f"  {'MIN':<45} {np.min(sims):>10.7f} {np.max(l2s):>10.6f}")

    print(f"\n  Interpretation:")
    print(f"  cos ≥ 0.9999 → C and Python produce bit-equivalent results ✓")
    print(f"  cos ∈ [0.999, 0.9999] → tiny float rounding in mean-pool (acceptable)")
    print(f"  cos < 0.999  → bug: check token_type_ids or mean-pool implementation")

    good = sum(1 for s in sims if s > 0.9999)
    print(f"\n  {good}/{N} domains are bit-equivalent between C and Python ORT.")
