/*
 * infer_arm64.c — MiniLM-L6 ONNX inference via ORT C API
 * =========================================================
 * Target: Bluefield 3 / ARM Cortex-A78 (AArch64)
 * Compiler: aarch64-linux-gnu-gcc or native gcc on BF3
 *
 * What this does:
 *   1. Loads the pre-tokenized domain inputs from a binary file
 *      (produced by 3_prepare_c_inputs.py so tokenization is identical)
 *   2. Runs OnnxRuntime C API inference (single-threaded, CPU EP)
 *   3. Computes mean-pool embedding → L2-normalizes
 *   4. Measures wall-clock latency with clock_gettime(CLOCK_MONOTONIC)
 *   5. Prints per-domain latency + embedding[0..3] for comparison with Python
 *
 * Build (native ARM64 on Bluefield 3):
 *   gcc -O2 -o infer_arm64 infer_arm64.c \
 *       -I/usr/local/include/onnxruntime \
 *       -L/usr/local/lib -lonnxruntime \
 *       -lm -Wl,-rpath,/usr/local/lib
 *
 * Build (cross-compile from x86):
 *   aarch64-linux-gnu-gcc -O2 -o infer_arm64 infer_arm64.c \
 *       -I${ORT_ARM64_ROOT}/include \
 *       -L${ORT_ARM64_ROOT}/lib -lonnxruntime \
 *       -lm -Wl,-rpath,/usr/local/lib
 *
 * Install ORT on Bluefield 3:
 *   pip3 install onnxruntime          # pulls the aarch64 wheel
 *   # OR download from:
 *   # https://github.com/microsoft/onnxruntime/releases
 *   # onnxruntime-linux-aarch64-<ver>.tgz
 *
 * Run:
 *   ./infer_arm64 models/minilm_int8.onnx inputs/domains.bin \
 *                 inputs/domains_meta.json
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* OnnxRuntime C API header — shipped with every ORT release */
#include "onnxruntime_c_api.h"


/* ── timing helper ────────────────────────────────────────────────────────── */
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

/* ── mean-pool: hidden[seq_len][embed_dim] × mask[seq_len] → emb[embed_dim] ─ */
static void mean_pool(const float *hidden,   /* [seq_len * embed_dim] */
                      const int64_t *mask,   /* [seq_len]             */
                      int seq_len, int embed_dim,
                      float *out             /* [embed_dim]           */) {
    memset(out, 0, embed_dim * sizeof(float));
    int count = 0;
    for (int s = 0; s < seq_len; s++) {
        if (mask[s] == 0) continue;
        count++;
        const float *row = hidden + s * embed_dim;
        for (int d = 0; d < embed_dim; d++)
            out[d] += row[d];
    }
    if (count > 0)
        for (int d = 0; d < embed_dim; d++)
            out[d] /= (float)count;
}

/* ── L2 normalize embedding in-place ─────────────────────────────────────── */
static void l2_normalize(float *v, int dim) {
    float norm = 0.0f;
    for (int d = 0; d < dim; d++) norm += v[d] * v[d];
    norm = sqrtf(norm) + 1e-9f;
    for (int d = 0; d < dim; d++) v[d] /= norm;
}

/* ── cosine similarity ────────────────────────────────────────────────────── */
static float cosine(const float *a, const float *b, int dim) {
    float dot = 0, na = 0, nb = 0;
    for (int d = 0; d < dim; d++) {
        dot += a[d] * b[d];
        na  += a[d] * a[d];
        nb  += b[d] * b[d];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-9f);
}

/* ── binary input format (written by 3_prepare_c_inputs.py) ─────────────── *
 *  Header: [n_domains: int32][seq_len: int32][embed_dim: int32]
 *  Per domain:
 *    domain_len: int32
 *    domain_str: char[domain_len]           (not null-terminated)
 *    input_ids:  int64[seq_len]
 *    attn_mask:  int64[seq_len]
 *    tok_types:  int64[seq_len]
 * ─────────────────────────────────────────────────────────────────────────── */
typedef struct {
    int      n_domains;
    int      seq_len;
    int      embed_dim;
    char   **domain_strs;
    int64_t *input_ids;    /* [n_domains * seq_len] */
    int64_t *attn_masks;   /* [n_domains * seq_len] */
    int64_t *tok_types;    /* [n_domains * seq_len] */
} InputData;

static InputData *load_inputs(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("open inputs"); return NULL; }

    InputData *d = calloc(1, sizeof(*d));
    fread(&d->n_domains, sizeof(int), 1, f);
    fread(&d->seq_len,   sizeof(int), 1, f);
    fread(&d->embed_dim, sizeof(int), 1, f);

    size_t total = (size_t)d->n_domains * d->seq_len;
    d->input_ids  = malloc(total * sizeof(int64_t));
    d->attn_masks = malloc(total * sizeof(int64_t));
    d->tok_types  = malloc(total * sizeof(int64_t));
    d->domain_strs = calloc(d->n_domains, sizeof(char *));

    for (int i = 0; i < d->n_domains; i++) {
        int dlen; fread(&dlen, sizeof(int), 1, f);
        d->domain_strs[i] = malloc(dlen + 1);
        fread(d->domain_strs[i], 1, dlen, f);
        d->domain_strs[i][dlen] = '\0';
        fread(d->input_ids  + i * d->seq_len, sizeof(int64_t), d->seq_len, f);
        fread(d->attn_masks + i * d->seq_len, sizeof(int64_t), d->seq_len, f);
        fread(d->tok_types  + i * d->seq_len, sizeof(int64_t), d->seq_len, f);
    }
    fclose(f);
    return d;
}

/* ── main ─────────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.onnx> <inputs.bin> [n_repeats]\n",
                argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    const char *input_path = argv[2];
    int n_repeats = (argc >= 4) ? atoi(argv[3]) : 200;

    /* ── 1. Load pre-tokenized inputs ─────────────────────────────────────── */
    printf("► Loading inputs from %s …\n", input_path);
    fflush(stdout);
    InputData *inp = load_inputs(input_path);
    if (!inp) return 1;
    printf("  %d domains, seq_len=%d, embed_dim=%d\n",
           inp->n_domains, inp->seq_len, inp->embed_dim);

    /* ── 2. Initialize OnnxRuntime environment ────────────────────────────── */
    const OrtApi *ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    OrtEnv *env;
    ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "dga_infer", &env);

    OrtSessionOptions *opts;
    ort->CreateSessionOptions(&opts);
    ort->SetIntraOpNumThreads(opts, 1);   /* single-thread: matches DPDK lcore */
    ort->SetInterOpNumThreads(opts, 1);
    ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);

    /* ── 3. Create session (loads and optimizes the ONNX graph) ───────────── */
    printf("► Loading ONNX model: %s …\n", model_path);
    fflush(stdout);
    double t_load0 = now_ms();
    OrtSession *session;
    OrtStatus *status = ort->CreateSession(env, model_path, opts, &session);
    if (status) {
        fprintf(stderr, "CreateSession error: %s\n", ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
        return 1;
    }
    printf("  Model load time: %.1f ms\n", now_ms() - t_load0);

    /* Allocator for output tensors */
    OrtAllocator *allocator;
    ort->GetAllocatorWithDefaultOptions(&allocator);

    /* ── 4. Prepare output embedding buffer ──────────────────────────────── */
    int embed_dim = inp->embed_dim;
    float *embedding = malloc(embed_dim * sizeof(float));

    /* ORT memory info for CPU tensors */
    OrtMemoryInfo *mem_info;
    ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);

    int seq_len   = inp->seq_len;
    int64_t shape_in[2] = { 1, seq_len };        /* batch=1 */
    const char *input_names[]  = {"input_ids", "attention_mask", "token_type_ids"};
    const char *output_names[] = {"last_hidden_state"};

    /* ── 5. Warm-up: 20 passes on first domain ───────────────────────────── */
    printf("► Warming up (20 passes) …\n");
    {
        int64_t *ids  = inp->input_ids;
        int64_t *mask = inp->attn_masks;
        int64_t *tt   = inp->tok_types;

        OrtValue *t_ids, *t_mask, *t_tt, *t_out;
        ort->CreateTensorWithDataAsOrtValue(mem_info, ids,  seq_len*sizeof(int64_t), shape_in, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_ids);
        ort->CreateTensorWithDataAsOrtValue(mem_info, mask, seq_len*sizeof(int64_t), shape_in, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_mask);
        ort->CreateTensorWithDataAsOrtValue(mem_info, tt,   seq_len*sizeof(int64_t), shape_in, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_tt);
        OrtValue *inputs[3] = {t_ids, t_mask, t_tt};

        for (int w = 0; w < 20; w++) {
            t_out = NULL;
            ort->Run(session, NULL, input_names, (const OrtValue *const *)inputs,
                     3, output_names, 1, &t_out);
            ort->ReleaseValue(t_out);
        }
        ort->ReleaseValue(t_ids);
        ort->ReleaseValue(t_mask);
        ort->ReleaseValue(t_tt);
    }

    /* ── 6. Per-domain latency measurement ───────────────────────────────── */
    printf("\n%s\n", "────────────────────────────────────────────────────────────────");
    printf("  %-45s %8s %8s %8s  emb[0..3]\n",
           "domain", "min_ms", "med_ms", "p95_ms");
    printf("%s\n", "────────────────────────────────────────────────────────────────");

    /* store per-domain embeddings for Python comparison */
    float *all_embeddings = malloc(inp->n_domains * embed_dim * sizeof(float));

    double *latencies = malloc(n_repeats * sizeof(double));

    for (int i = 0; i < inp->n_domains; i++) {
        int64_t *ids  = inp->input_ids  + i * seq_len;
        int64_t *mask = inp->attn_masks + i * seq_len;
        int64_t *tt   = inp->tok_types  + i * seq_len;

        /* Build input tensors (zero-copy: ORT uses our buffer directly) */
        OrtValue *t_ids, *t_mask, *t_tt;
        ort->CreateTensorWithDataAsOrtValue(mem_info, ids,  seq_len*sizeof(int64_t), shape_in, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_ids);
        ort->CreateTensorWithDataAsOrtValue(mem_info, mask, seq_len*sizeof(int64_t), shape_in, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_mask);
        ort->CreateTensorWithDataAsOrtValue(mem_info, tt,   seq_len*sizeof(int64_t), shape_in, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_tt);
        OrtValue *inputs[3] = {t_ids, t_mask, t_tt};

        /* measure n_repeats runs */
        for (int r = 0; r < n_repeats; r++) {
            OrtValue *t_out = NULL;
            double t0 = now_ms();
            ort->Run(session, NULL, input_names,
                     (const OrtValue *const *)inputs, 3,
                     output_names, 1, &t_out);
            latencies[r] = now_ms() - t0;

            /* on last repeat: extract embedding */
            if (r == n_repeats - 1) {
                float *hidden;
                ort->GetTensorMutableData(t_out, (void **)&hidden);
                /* last_hidden_state shape: [1, seq_len, embed_dim] */
                mean_pool(hidden, mask, seq_len, embed_dim,
                          all_embeddings + i * embed_dim);
                l2_normalize(all_embeddings + i * embed_dim, embed_dim);
            }
            ort->ReleaseValue(t_out);
        }

        /* compute statistics */
        /* simple insertion sort for percentiles (n_repeats ≤ 500 is fine) */
        double tmp[n_repeats];
        memcpy(tmp, latencies, n_repeats * sizeof(double));
        for (int a = 1; a < n_repeats; a++) {
            double key = tmp[a]; int b = a - 1;
            while (b >= 0 && tmp[b] > key) { tmp[b+1] = tmp[b]; b--; }
            tmp[b+1] = key;
        }
        double lat_min = tmp[0];
        double lat_med = tmp[n_repeats / 2];
        double lat_p95 = tmp[(int)(0.95 * n_repeats)];

        float *emb = all_embeddings + i * embed_dim;
        printf("  %-45s %8.3f %8.3f %8.3f  [%.4f %.4f %.4f %.4f]\n",
               inp->domain_strs[i],
               lat_min, lat_med, lat_p95,
               emb[0], emb[1], emb[2], emb[3]);

        ort->ReleaseValue(t_ids);
        ort->ReleaseValue(t_mask);
        ort->ReleaseValue(t_tt);
    }

    /* ── 7. Save embeddings for Python cross-validation ─────────────────── */
    FILE *ef = fopen("outputs/c_embeddings.bin", "wb");
    if (ef) {
        fwrite(&inp->n_domains, sizeof(int), 1, ef);
        fwrite(&embed_dim,      sizeof(int), 1, ef);
        fwrite(all_embeddings,  sizeof(float), inp->n_domains * embed_dim, ef);
        fclose(ef);
        printf("\n► Embeddings saved → outputs/c_embeddings.bin\n");
    }

    /* ── 8. cleanup ─────────────────────────────────────────────────────── */
    free(latencies);
    free(embedding);
    free(all_embeddings);
    ort->ReleaseMemoryInfo(mem_info);
    ort->ReleaseSession(session);
    ort->ReleaseSessionOptions(opts);
    ort->ReleaseEnv(env);

    printf("✓ Done\n");
    return 0;
}
