"""
Step 2 — Offline benchmark: PyTorch vs ONNX-FP32 vs ONNX-INT8
==============================================================
Measures:
  - Per-domain inference latency (warm + cold runs)
  - Embedding cosine similarity (numerical drift from quantization)
  - Throughput at batch sizes 1, 8, 32

Test domains: benign mix (no malicious needed — we're testing inference time,
not detection accuracy). Covers short/long/hyphenated/IDN patterns.

Run: python3 2_benchmark_python.py
"""

import json
import time
import statistics
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import onnxruntime as ort

# ── config ────────────────────────────────────────────────────────────────────
MODELS_DIR  = Path("models")
ONNX_FP32   = MODELS_DIR / "minilm.onnx"
ONNX_INT8   = MODELS_DIR / "minilm_int8.onnx"
N_WARMUP    = 20     # warm-up iterations (JIT / ORT graph optimization settle)
N_BENCH     = 200    # measured iterations
MAX_SEQ_LEN = 64

# ── test domains (realistic distribution, no malicious needed) ─────────────────
# Short legit, long legit, hyphenated, CDN patterns, ambiguous-length
TEST_DOMAINS = [
    # ── clearly benign (short, common TLD) ──
    "google.com", "github.com", "amazon.com", "facebook.com", "twitter.com",
    "netflix.com", "reddit.com", "wikipedia.org", "stackoverflow.com",
    "cloudflare.com", "fastly.net", "akamaiedge.net", "cdn.jsdelivr.net",
    # ── longer / hyphenated ──
    "mail.google.com", "api.github.com", "docs.microsoft.com",
    "security.ubuntu.com", "archive.ubuntu.com",
    "fonts.googleapis.com", "static.cloudflareinsights.com",
    "edge-star-shv-01-iad3.facebook.com",
    # ── corporate / enterprise patterns ──
    "login.microsoftonline.com", "s3.amazonaws.com",
    "ec2-54-234-12-1.compute.amazonaws.com",
    "storage.googleapis.com", "accounts.google.com",
    # ── numeric / version-like (neutral entropy) ──
    "1e100.net", "v4.api.example.com", "cdn2.example.co.uk",
    # ── longer subdomains (stress test seq_len) ──
    "very-long-subdomain-name-for-testing-purposes.example.com",
    "another-quite-long-subdomain.with-hyphens.and-dots.example.org",
]

# ── helpers ────────────────────────────────────────────────────────────────────
def mean_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean-pool last_hidden_state over non-padding tokens."""
    m   = mask[..., None].astype(np.float32)       # [B, S, 1]
    emb = (hidden * m).sum(1) / (m.sum(1) + 1e-9)  # [B, 384]
    return emb

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9
    nb = np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9
    return ((a / na) * (b / nb)).sum(-1)

def format_time(ms: float) -> str:
    if ms < 1:
        return f"{ms*1000:.1f} µs"
    return f"{ms:.3f} ms"

# ── load resources ─────────────────────────────────────────────────────────────
print("► Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR / "tokenizer"))

print("► Loading PyTorch model …")
pt_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
pt_model.eval()

print("► Loading ONNX sessions …")
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 1    # single-thread → matches SmartNIC lcore
sess_opts.inter_op_num_threads = 1
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

sess_fp32 = ort.InferenceSession(str(ONNX_FP32), sess_options=sess_opts,
                                  providers=["CPUExecutionProvider"])
sess_int8 = ort.InferenceSession(str(ONNX_INT8), sess_options=sess_opts,
                                  providers=["CPUExecutionProvider"])

# ── inference functions ────────────────────────────────────────────────────────
def infer_pytorch(domains: list[str]) -> np.ndarray:
    enc = tokenizer(domains, padding=True, truncation=True,
                    max_length=MAX_SEQ_LEN, return_tensors="pt")
    with torch.no_grad():
        out = pt_model(**enc)
    mask = enc["attention_mask"].numpy()
    return mean_pool(out.last_hidden_state.numpy(), mask)

def infer_onnx(sess: ort.InferenceSession, domains: list[str]) -> np.ndarray:
    enc = tokenizer(domains, padding=True, truncation=True,
                    max_length=MAX_SEQ_LEN, return_tensors="pt")
    feeds = {
        "input_ids":      enc["input_ids"].numpy().astype(np.int64),
        "attention_mask": enc["attention_mask"].numpy().astype(np.int64),
        "token_type_ids": enc.get("token_type_ids",
                          torch.zeros_like(enc["input_ids"])
                          ).numpy().astype(np.int64),
    }
    hidden = sess.run(["last_hidden_state"], feeds)[0]  # [B, S, 384]
    return mean_pool(hidden, feeds["attention_mask"])

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A — Single-domain latency (batch=1, most relevant for SmartNIC)
# This is what DPDK sees: one domain dequeued from rte_ring → infer → score
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("EXPERIMENT A — Single-domain latency  (batch=1, N=200)")
print("═"*62)

SINGLE_DOMAIN = ["google.com"]

for label, fn in [
    ("PyTorch FP32",  lambda: infer_pytorch(SINGLE_DOMAIN)),
    ("ONNX FP32",     lambda: infer_onnx(sess_fp32, SINGLE_DOMAIN)),
    ("ONNX INT8",     lambda: infer_onnx(sess_int8, SINGLE_DOMAIN)),
]:
    # warm-up
    for _ in range(N_WARMUP):
        fn()

    times = []
    for _ in range(N_BENCH):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)  # ms

    p50  = statistics.median(times)
    p95  = sorted(times)[int(0.95 * len(times))]
    p99  = sorted(times)[int(0.99 * len(times))]
    mean = statistics.mean(times)
    print(f"  {label:<18}  mean={format_time(mean)}  "
          f"p50={format_time(p50)}  p95={format_time(p95)}  p99={format_time(p99)}")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — Batch latency sweep (how well does batching help?)
# Relevant if you accumulate domains in rte_ring before flushing to inference
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("EXPERIMENT B — Batch size sweep  (ONNX INT8, N=100)")
print("═"*62)
print(f"  {'batch':>6}  {'total ms':>10}  {'per-domain ms':>14}  {'domains/sec':>12}")
print("  " + "-"*46)

for bs in [1, 2, 4, 8, 16, 32]:
    batch = (TEST_DOMAINS * ((bs // len(TEST_DOMAINS)) + 1))[:bs]
    for _ in range(N_WARMUP):
        infer_onnx(sess_int8, batch)
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        infer_onnx(sess_int8, batch)
        times.append((time.perf_counter() - t0) * 1000)
    total = statistics.median(times)
    per_d = total / bs
    dps   = 1000 / per_d
    print(f"  {bs:>6}  {total:>10.3f}  {per_d:>14.3f}  {dps:>12.0f}")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT C — Per-domain latency (batch=1) across all test domains
# Shows variance due to domain length (seq_len changes tokenized length)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("EXPERIMENT C — Per-domain latency variance  (ONNX INT8, batch=1)")
print("═"*62)
print(f"  {'domain':<45} {'tok_len':>7} {'med ms':>8}")
print("  " + "-"*62)

for d in TEST_DOMAINS:
    enc   = tokenizer([d], padding=False, truncation=True, max_length=MAX_SEQ_LEN)
    tlen  = len(enc["input_ids"][0])
    for _ in range(10):
        infer_onnx(sess_int8, [d])
    times = []
    for _ in range(60):
        t0 = time.perf_counter()
        infer_onnx(sess_int8, [d])
        times.append((time.perf_counter() - t0) * 1000)
    med = statistics.median(times)
    print(f"  {d:<45} {tlen:>7} {med:>8.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT D — Numerical drift: PyTorch vs ONNX FP32 vs ONNX INT8
# Critical: ensure quantization doesn't shift embeddings meaningfully
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("EXPERIMENT D — Embedding drift (cosine similarity to PyTorch ground truth)")
print("═"*62)
print(f"  {'domain':<45} {'PT↔FP32':>9} {'PT↔INT8':>9} {'FP32↔INT8':>11}")
print("  " + "-"*76)

all_pt   = []
all_fp32 = []
all_int8 = []
for d in TEST_DOMAINS:
    pt_e   = infer_pytorch([d])
    fp32_e = infer_onnx(sess_fp32, [d])
    int8_e = infer_onnx(sess_int8, [d])
    all_pt.append(pt_e[0]);  all_fp32.append(fp32_e[0]);  all_int8.append(int8_e[0])
    c1 = cosine_sim(pt_e, fp32_e)[0]
    c2 = cosine_sim(pt_e, int8_e)[0]
    c3 = cosine_sim(fp32_e, int8_e)[0]
    print(f"  {d:<45} {c1:>9.6f} {c2:>9.6f} {c3:>11.6f}")

all_pt   = np.stack(all_pt)
all_fp32 = np.stack(all_fp32)
all_int8 = np.stack(all_int8)
print("  " + "-"*76)
avg_c1 = cosine_sim(all_pt, all_fp32).mean()
avg_c2 = cosine_sim(all_pt, all_int8).mean()
avg_c3 = cosine_sim(all_fp32, all_int8).mean()
print(f"  {'AVERAGE':<45} {avg_c1:>9.6f} {avg_c2:>9.6f} {avg_c3:>11.6f}")

l2_fp32 = np.linalg.norm(all_pt - all_fp32, axis=-1).mean()
l2_int8 = np.linalg.norm(all_pt - all_int8, axis=-1).mean()
print(f"\n  Mean L2 drift  PT→FP32: {l2_fp32:.6f}   PT→INT8: {l2_int8:.6f}")
print(f"  INT8 cosine similarity ≥0.999 means quantization is safe for deployment.")

# ── save raw timing data for plots / paper tables ─────────────────────────────
print("\n► Saving timing data → models/benchmark_results.json …")
results = {
    "domains": TEST_DOMAINS,
    "pt_fp32_single_latency_ms": None,   # re-collect below
    "onnx_fp32_single_latency_ms": None,
    "onnx_int8_single_latency_ms": None,
}

timing_data = {}
for label, fn in [
    ("pytorch_fp32", lambda: infer_pytorch(["google.com"])),
    ("onnx_fp32",    lambda: infer_onnx(sess_fp32, ["google.com"])),
    ("onnx_int8",    lambda: infer_onnx(sess_int8, ["google.com"])),
]:
    for _ in range(N_WARMUP):
        fn()
    times = [(time.perf_counter(), fn())[0] for _ in range(N_BENCH)]
    raw = []
    for _ in range(N_BENCH):
        t0 = time.perf_counter()
        fn()
        raw.append((time.perf_counter() - t0) * 1000)
    timing_data[label] = {
        "mean_ms":   statistics.mean(raw),
        "median_ms": statistics.median(raw),
        "p95_ms":    sorted(raw)[int(0.95 * len(raw))],
        "p99_ms":    sorted(raw)[int(0.99 * len(raw))],
        "min_ms":    min(raw),
        "max_ms":    max(raw),
    }

with open(MODELS_DIR / "benchmark_results.json", "w") as f:
    json.dump(timing_data, f, indent=2)

print("\n✓ Benchmark complete.")
print("\nKey takeaways for paper:")
pt_med   = timing_data["pytorch_fp32"]["median_ms"]
fp32_med = timing_data["onnx_fp32"]["median_ms"]
int8_med = timing_data["onnx_int8"]["median_ms"]
print(f"  PyTorch FP32 median:  {pt_med:.3f} ms")
print(f"  ONNX FP32 median:     {fp32_med:.3f} ms  ({pt_med/fp32_med:.1f}× vs PyTorch)")
print(f"  ONNX INT8 median:     {int8_med:.3f} ms  ({pt_med/int8_med:.1f}× vs PyTorch)")
print(f"  Avg cosine(PT,INT8):  {avg_c2:.6f}  (1.0 = identical)")
