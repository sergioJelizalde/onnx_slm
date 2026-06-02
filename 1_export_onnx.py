"""
Step 1 — Export all-MiniLM-L6-v2 to ONNX (FP32 + INT8 quantized)
=================================================================
What this does:
  - Downloads sentence-transformers/all-MiniLM-L6-v2 (22M params)
  - Exports the BERT backbone to ONNX opset 17
  - Quantizes to INT8 (dynamic) for ARM deployment
  - Saves tokenizer vocab + config for use in C inference

Output files:
  models/minilm.onnx          - FP32 ONNX graph
  models/minilm_int8.onnx     - INT8 quantized (for Bluefield 3)
  models/tokenizer/           - HuggingFace tokenizer files (used by Python)
  models/vocab.txt            - Raw vocab for C tokenizer

Architecture reminder:
  Input: [input_ids, attention_mask, token_type_ids]  shape: [B, seq_len]
  Output: last_hidden_state shape: [B, seq_len, 384]
          We take mean-pool over seq_len → [B, 384] as domain embedding
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from onnxruntime.quantization import quantize_dynamic, QuantType

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ── paths ─────────────────────────────────────────────────────────────────────
MODEL_ID  = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR   = Path("models")
OUT_DIR.mkdir(exist_ok=True)
TOK_DIR   = OUT_DIR / "tokenizer"
TOK_DIR.mkdir(exist_ok=True)
ONNX_FP32 = OUT_DIR / "minilm.onnx"
ONNX_INT8 = OUT_DIR / "minilm_int8.onnx"

# ── 1. Load model & tokenizer ─────────────────────────────────────────────────
print("► Loading model …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModel.from_pretrained(MODEL_ID)
model.eval()

# Save tokenizer files (Python side uses these; C uses vocab.txt directly)
tokenizer.save_pretrained(str(TOK_DIR))
print(f"  Tokenizer saved → {TOK_DIR}")

# Also dump a plain vocab.txt for the C wordpiece tokenizer
with open(OUT_DIR / "vocab.txt", "w", encoding="utf-8") as f:
    for token, idx in sorted(tokenizer.vocab.items(), key=lambda x: x[1]):
        f.write(token + "\n")
print(f"  vocab.txt → {OUT_DIR / 'vocab.txt'} ({len(tokenizer.vocab)} tokens)")

# ── 2. Build a representative dummy input ─────────────────────────────────────
# Max domain length: 63 chars + subword overhead ≈ 32 tokens is generous
# We export with dynamic axes so any seq_len works at runtime.
DUMMY_DOMAINS = ["google.com", "xkqpzmvtlrjbd.ru", "translateincoming.com"]
enc = tokenizer(DUMMY_DOMAINS, padding=True, truncation=True,
                max_length=64, return_tensors="pt")

input_ids      = enc["input_ids"]        # [3, seq_len]
attention_mask = enc["attention_mask"]   # [3, seq_len]
token_type_ids = enc.get("token_type_ids",
                          torch.zeros_like(input_ids))  # MiniLM may omit this

print(f"\n► Dummy input shape: {input_ids.shape}  "
      f"(batch=3, seq_len={input_ids.shape[1]})")

# ── 3. Verify PyTorch forward pass ────────────────────────────────────────────
with torch.no_grad():
    pt_out = model(input_ids=input_ids,
                   attention_mask=attention_mask,
                   token_type_ids=token_type_ids)

last_hidden = pt_out.last_hidden_state   # [3, seq_len, 384]

# Mean-pool (mask padding tokens out)
mask_exp = attention_mask.unsqueeze(-1).float()  # [3, seq_len, 1]
pt_emb   = (last_hidden * mask_exp).sum(1) / mask_exp.sum(1)  # [3, 384]
print(f"  PyTorch embedding shape: {pt_emb.shape}")
print(f"  Sample norm (domain 0): {pt_emb[0].norm().item():.4f}")

# ── 4. Export to ONNX ─────────────────────────────────────────────────────────
print(f"\n► Exporting FP32 ONNX → {ONNX_FP32} …")

torch.onnx.export(
    model,
    args=(input_ids, attention_mask, token_type_ids),
    f=str(ONNX_FP32),
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        # Both batch size and sequence length are dynamic
        "input_ids":        {0: "batch", 1: "seq_len"},
        "attention_mask":   {0: "batch", 1: "seq_len"},
        "token_type_ids":   {0: "batch", 1: "seq_len"},
        "last_hidden_state":{0: "batch", 1: "seq_len"},
    },
    opset_version=17,
    do_constant_folding=True,   # folds BatchNorm, Dropout, etc. at export time
    verbose=False,
    dynamo=False,
)
size_fp32 = ONNX_FP32.stat().st_size / 1e6
print(f"  FP32 model size: {size_fp32:.1f} MB")

# ── 5. INT8 dynamic quantization (ARM-friendly) ───────────────────────────────
# Dynamic quantization: weights quantized at export; activations at runtime.
# This avoids a calibration dataset and is safe for production.
# For ARM A78: OnnxRuntime selects NEON-optimized INT8 GEMM kernels automatically.
print(f"\n► Quantizing INT8 → {ONNX_INT8} …")
quantize_dynamic(
    model_input=str(ONNX_FP32),
    model_output=str(ONNX_INT8),
    weight_type=QuantType.QInt8,
    # Quantize all MatMul and Gemm nodes (the bulk of transformer compute)
    op_types_to_quantize=["MatMul"],
    per_channel=True,     # per-tensor is faster on ARM; per-channel is more accurate
    reduce_range=False,    # reduce_range=True for x86 VNNI; False for ARM
)
size_int8 = ONNX_INT8.stat().st_size / 1e6
print(f"  INT8 model size: {size_int8:.1f} MB  "
      f"({100*(1-size_int8/size_fp32):.0f}% reduction)")

# ── 6. Quick ONNX verify ──────────────────────────────────────────────────────
import onnxruntime as ort

def run_onnx(onnx_path, enc_dict):
    sess = ort.InferenceSession(str(onnx_path),
              providers=["CPUExecutionProvider"])
    feeds = {
        "input_ids":      enc_dict["input_ids"].numpy().astype(np.int64),
        "attention_mask": enc_dict["attention_mask"].numpy().astype(np.int64),
        "token_type_ids": enc_dict.get("token_type_ids",
                          torch.zeros_like(enc_dict["input_ids"])
                          ).numpy().astype(np.int64),
    }
    hidden = sess.run(["last_hidden_state"], feeds)[0]   # [B, S, 384]
    mask   = feeds["attention_mask"][..., None]           # [B, S, 1]
    emb    = (hidden * mask).sum(1) / mask.sum(1)        # [B, 384]
    return emb

ort_fp32_emb = run_onnx(ONNX_FP32, enc)
ort_int8_emb = run_onnx(ONNX_INT8, enc)

# Cosine similarity between PyTorch and ONNX outputs (should be ~1.0)
def cosine(a, b):
    a = np.array(a).flatten().astype(np.float64)
    b = np.array(b).flatten().astype(np.float64)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

pt_np = pt_emb.numpy()
print("\n► Numerical equivalence check:")
print("  domain                    PT↔ONNX_FP32  PT↔ONNX_INT8")
print("  " + "-"*52)
for i, d in enumerate(DUMMY_DOMAINS):
    c1 = cosine(pt_np[i], ort_fp32_emb[i])
    c2 = cosine(pt_np[i], ort_int8_emb[i])
    print(f"  {d:<26} {c1:.6f}      {c2:.6f}")

# ── 7. Save metadata for C loader ────────────────────────────────────────────
meta = {
    "model_id":     MODEL_ID,
    "embed_dim":    384,
    "max_seq_len":  64,
    "pad_token_id": tokenizer.pad_token_id,
    "cls_token_id": tokenizer.cls_token_id,
    "sep_token_id": tokenizer.sep_token_id,
    "unk_token_id": tokenizer.unk_token_id,
    "vocab_size":   len(tokenizer.vocab),
    "onnx_fp32":    str(ONNX_FP32),
    "onnx_int8":    str(ONNX_INT8),
}
with open(OUT_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"\n► Metadata → {OUT_DIR / 'meta.json'}")
print("\n✓ Export complete. Files in ./models/")
print(f"  FP32:  {ONNX_FP32}  ({size_fp32:.1f} MB)")
print(f"  INT8:  {ONNX_INT8}  ({size_int8:.1f} MB)")
