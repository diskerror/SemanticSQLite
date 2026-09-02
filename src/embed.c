// embed.c — EMBED(text) via either an ONNX model (default, matching Ragger's
// pipeline exactly) or a GGUF model via libllama. Backend is selected by the
// `embedding_embedder` config key ("onnx" or "llama").
//
// Config keys consulted by EMBED():
//   embedding_embedder     — "onnx" (default) or "llama"
//   embedding_model        — path to the model (ONNX: directory containing
//                             model.onnx + tokenizer.json; llama: path to a
//                             single .gguf file)
//   embedding_dims         — target output dimension, 1..4096 (optional;
//                             omit/0 to use the model's native n_embd).
//                             Larger than native = zero-padded; smaller =
//                             truncated (Matryoshka-style; not re-trained,
//                             just a slice — fine for experimentation, not
//                             claimed to preserve quality at all sizes).
//   embedding_vector_type  — "f16" (default), "f32", "bf16", or "int8"
//   embedding_offset       — byte offset prepended to the blob (default 0;
//                             the offset bytes are zero-filled, allowing the
//                             caller to reserve space for a header that an
//                             external tool writes separately)
//   embedding_skip_renorm  — "1"/"true"/"yes" to skip L2 normalization
//
// The model is loaded lazily on first EMBED() call and cached for the
// process lifetime, keyed by path — calling SEMEXT_SET('embedding_model', X)
// with a different path swaps the cached model on the next EMBED() call.

#include "embed.h"
#include "embed_llama.h"
#ifdef SEMEXT_HAVE_ONNX
#include "embed_onnx.h"
#endif
#include "EmbeddingCodecCapi.h"

#include <llama.h>
#include <sqlite3.h>

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* semext_config table — key/value store standing in for PRAGMAs       */
/* ------------------------------------------------------------------ */
static int ensure_config_table(sqlite3 *db) {
    return sqlite3_exec(db,
        "CREATE TABLE IF NOT EXISTS semext_config ("
        "  key TEXT PRIMARY KEY,"
        "  value TEXT"
        ")", NULL, NULL, NULL);
}

// Reads a config value into a newly malloc'd string (caller frees), or
// NULL if the key is absent. Returns SQLITE_OK/error code via *rc if given.
static char *config_get(sqlite3 *db, const char *key) {
    sqlite3_stmt *stmt = NULL;
    char *result = NULL;
    if (sqlite3_prepare_v2(db, "SELECT value FROM semext_config WHERE key = ?", -1, &stmt, NULL) != SQLITE_OK) {
        return NULL;
    }
    sqlite3_bind_text(stmt, 1, key, -1, SQLITE_STATIC);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *v = (const char *)sqlite3_column_text(stmt, 0);
        if (v) {
            size_t len = strlen(v);
            result = (char *)malloc(len + 1);
            if (result) memcpy(result, v, len + 1);
        }
    }
    sqlite3_finalize(stmt);
    return result;
}

static void config_set(sqlite3 *db, const char *key, const char *value) {
    ensure_config_table(db);
    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(db,
            "INSERT INTO semext_config (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            -1, &stmt, NULL) != SQLITE_OK) {
        return;
    }
    sqlite3_bind_text(stmt, 1, key, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, value, -1, SQLITE_STATIC);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

static void semext_set_func(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2 || sqlite3_value_type(argv[0]) == SQLITE_NULL) {
        sqlite3_result_error(ctx, "SEMEXT_SET(key, value): key required", -1);
        return;
    }
    sqlite3 *db = sqlite3_context_db_handle(ctx);
    const char *key = (const char *)sqlite3_value_text(argv[0]);
    const char *val = (sqlite3_value_type(argv[1]) == SQLITE_NULL)
                     ? "" : (const char *)sqlite3_value_text(argv[1]);
    config_set(db, key, val);
    sqlite3_result_text(ctx, val, -1, SQLITE_TRANSIENT);
}

static void semext_get_func(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 1 || sqlite3_value_type(argv[0]) == SQLITE_NULL) {
        sqlite3_result_null(ctx);
        return;
    }
    sqlite3 *db = sqlite3_context_db_handle(ctx);
    ensure_config_table(db);
    const char *key = (const char *)sqlite3_value_text(argv[0]);
    char *val = config_get(db, key);
    if (!val) { sqlite3_result_null(ctx); return; }
    sqlite3_result_text(ctx, val, -1, SQLITE_TRANSIENT);
    free(val);
}

/* ------------------------------------------------------------------ */
/* libllama model cache — one model loaded at a time, keyed by path.   */
/* Not thread-safe; fine for a single-connection CLI shell.            */
/* ------------------------------------------------------------------ */
typedef struct {
    char              *path;
    struct llama_model   *model;
    struct llama_context *ctx;
    int                n_embd;
} embed_model_cache;

static embed_model_cache g_cache = {0};
static int g_backend_initialized = 0;

static void unload_cached_model(void) {
    if (g_cache.ctx)   { llama_free(g_cache.ctx);        g_cache.ctx = NULL; }
    if (g_cache.model) { llama_model_free(g_cache.model); g_cache.model = NULL; }
    free(g_cache.path);
    g_cache.path = NULL;
    g_cache.n_embd = 0;
}

void semext_llama_shutdown(void) {
    unload_cached_model();
    if (g_backend_initialized) {
        llama_backend_free();
        g_backend_initialized = 0;
    }
}

void semext_embed_shutdown(void) {
    semext_llama_shutdown();
#ifdef SEMEXT_HAVE_ONNX
    semext_onnx_shutdown();
#endif
}

int semext_llama_load(const char *path, const char **errmsg) {
    *errmsg = NULL;
    if (!path || path[0] == '\0') {
        *errmsg = "embedding_model not set — call SEMEXT_SET('embedding_model', '/path/to/model.gguf') first";
        return -1;
    }
    if (g_cache.path && strcmp(g_cache.path, path) == 0 && g_cache.model) {
        return 0;  // already loaded
    }

    unload_cached_model();

    if (!g_backend_initialized) {
        llama_backend_init();
        g_backend_initialized = 1;
    }

    struct llama_model_params mparams = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(path, mparams);
    if (!model) {
        *errmsg = "failed to load GGUF model (bad path or unsupported file)";
        return -1;
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.embeddings = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    cparams.n_ubatch = 2048;
    cparams.n_batch  = 2048;

    struct llama_context *lctx = llama_init_from_model(model, cparams);
    if (!lctx) {
        llama_model_free(model);
        *errmsg = "failed to create llama_context for model";
        return -1;
    }

    size_t path_len = strlen(path);
    g_cache.path = (char *)malloc(path_len + 1);
    if (g_cache.path) memcpy(g_cache.path, path, path_len + 1);
    g_cache.model = model;
    g_cache.ctx = lctx;
    g_cache.n_embd = llama_model_n_embd(model);

    // atexit cleanup is registered centrally in embed_func() after
    // any successful backend load — not here — so the LIFO ordering
    // works for both llama-only and ONNX-only usage.
    return 0;
}

int semext_llama_encode(const char *text, int text_len,
                        float **out, int *out_dims) {
    *out = NULL;
    *out_dims = 0;
    if (!g_cache.model || !g_cache.ctx) return -1;

    const struct llama_vocab *vocab = llama_model_get_vocab(g_cache.model);
    int n_embd = g_cache.n_embd;

    int n_tokens_max = text_len + 8;
    llama_token *tokens = (llama_token *)malloc(sizeof(llama_token) * (size_t)n_tokens_max);
    if (!tokens) return -1;

    int n_tokens = llama_tokenize(vocab, text, text_len, tokens, n_tokens_max, true, true);
    if (n_tokens < 0) {
        n_tokens_max = -n_tokens + 8;
        llama_token *retry = (llama_token *)realloc(tokens, sizeof(llama_token) * (size_t)n_tokens_max);
        if (!retry) { free(tokens); return -1; }
        tokens = retry;
        n_tokens = llama_tokenize(vocab, text, text_len, tokens, n_tokens_max, true, true);
    }
    if (n_tokens <= 0) { free(tokens); return -1; }

    llama_memory_clear(llama_get_memory(g_cache.ctx), true);
    struct llama_batch batch = llama_batch_get_one(tokens, n_tokens);
    int rc = llama_decode(g_cache.ctx, batch);
    free(tokens);
    if (rc != 0) return -1;

    const float *embd = llama_get_embeddings_seq(g_cache.ctx, 0);
    if (!embd) embd = llama_get_embeddings_ith(g_cache.ctx, 0);
    if (!embd) return -1;

    // L2-normalize (matches Ragger's convention).
    float *result = (float *)malloc(sizeof(float) * (size_t)n_embd);
    if (!result) return -1;
    double norm = 0.0;
    for (int i = 0; i < n_embd; i++) norm += (double)embd[i] * (double)embd[i];
    norm = sqrt(norm);
    float inv = (norm > 1e-12) ? (float)(1.0 / norm) : 0.0f;
    for (int i = 0; i < n_embd; i++) result[i] = embd[i] * inv;

    *out = result;
    *out_dims = n_embd;
    return 0;
}

// Vector type identifiers — must match ext_functions.c's enum and
// diskerror_vector_type in c_lib's EmbeddingCodecCapi.h.
enum embed_vec_type { EVT_F32 = 0, EVT_F16 = 1, EVT_BF16 = 2, EVT_INT8 = 3 };

static enum embed_vec_type parse_embed_vec_type(const char *s) {
    if (!s || s[0] == '\0') return EVT_F16;
    char c0 = (s[0] >= 'A' && s[0] <= 'Z') ? (char)(s[0] - 'A' + 'a') : s[0];
    if (c0 == 'f') {
        if (s[1] == '3' && s[2] == '2') return EVT_F32;
        return EVT_F16;
    }
    if (c0 == 'b') return EVT_BF16;
    if (c0 == 'i' || c0 == 'q') return EVT_INT8;
    return EVT_F16;
}

/* ------------------------------------------------------------------ */
/* Embedder backend identifiers                                        */
/* ------------------------------------------------------------------ */
enum embed_backend { EB_ONNX = 0, EB_LLAMA = 1 };

static enum embed_backend parse_backend(const char *s) {
    if (!s || s[0] == '\0') {
#ifdef SEMEXT_HAVE_ONNX
        return EB_ONNX;
#else
        return EB_LLAMA;
#endif
    }
    char c0 = (s[0] >= 'A' && s[0] <= 'Z') ? (char)(s[0] - 'A' + 'a') : s[0];
    if (c0 == 'l' || c0 == 'g') return EB_LLAMA;  // llama / gguf
    return EB_ONNX;
}

/* ------------------------------------------------------------------ */
/* EMBED(text) -> BLOB embedding, per semext_config settings           */
/* ------------------------------------------------------------------ */
static void embed_func(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 1 || sqlite3_value_type(argv[0]) == SQLITE_NULL) {
        sqlite3_result_null(ctx);
        return;
    }
    const char *text = (const char *)sqlite3_value_text(argv[0]);
    int text_len = sqlite3_value_bytes(argv[0]);

    sqlite3 *db = sqlite3_context_db_handle(ctx);
    ensure_config_table(db);

    char *embedder_str = config_get(db, "embedding_embedder");
    char *model_path   = config_get(db, "embedding_model");
    char *dims_str     = config_get(db, "embedding_dims");
    char *vtype_str    = config_get(db, "embedding_vector_type");
    char *offset_str   = config_get(db, "embedding_offset");
    char *skip_renorm_str = config_get(db, "embedding_skip_renorm");

    enum embed_backend backend = parse_backend(embedder_str);
    free(embedder_str);

    // Load the model via the selected backend.
    const char *errmsg = NULL;
    int load_rc;
    switch (backend) {
#ifdef SEMEXT_HAVE_ONNX
        case EB_ONNX:
            load_rc = semext_onnx_load(model_path, &errmsg);
            break;
#endif
        case EB_LLAMA:
        default:
            load_rc = semext_llama_load(model_path, &errmsg);
            break;
    }
    if (load_rc != 0) {
        if (!errmsg) errmsg = "failed to load embedding model";
        sqlite3_result_error(ctx, errmsg, -1);
        free(model_path); free(dims_str); free(vtype_str);
        free(offset_str); free(skip_renorm_str);
        return;
    }
    free(model_path);

    // Ensure cleanup runs at exit for whichever backend(s) were loaded.
    // Must be registered AFTER any llama model load (atexit LIFO ordering
    // vs ggml's Metal cleanup — see the comment in semext_llama_load).
    // For ONNX-only usage, the llama backend was never touched so there's
    // no ggml ordering constraint, but we still need to free the ONNX env.
    {
        static int atexit_registered = 0;
        if (!atexit_registered) {
            atexit(semext_embed_shutdown);
            atexit_registered = 1;
        }
    }

    int target_dims = 0;
    if (dims_str && dims_str[0]) {
        target_dims = atoi(dims_str);
        if (target_dims < 0) target_dims = 0;
        if (target_dims > 4096) target_dims = 4096;
    }
    free(dims_str);

    enum embed_vec_type vtype = parse_embed_vec_type(vtype_str);
    free(vtype_str);

    int blob_offset = 0;
    if (offset_str && offset_str[0]) {
        blob_offset = atoi(offset_str);
        if (blob_offset < 0) blob_offset = 0;
    }
    free(offset_str);

    int skip_renorm = 0;
    if (skip_renorm_str) {
        skip_renorm = (skip_renorm_str[0] == '1' ||
                       skip_renorm_str[0] == 't' || skip_renorm_str[0] == 'T' ||
                       skip_renorm_str[0] == 'y' || skip_renorm_str[0] == 'Y');
        free(skip_renorm_str);
    }

    // Encode via the selected backend. The backend returns a normalized f32
    // vector (both backends L2-normalize internally).
    float *raw_emb = NULL;
    int raw_dims = 0;
    int enc_rc;
    switch (backend) {
#ifdef SEMEXT_HAVE_ONNX
        case EB_ONNX:
            enc_rc = semext_onnx_encode(text, text_len, &raw_emb, &raw_dims);
            break;
#endif
        case EB_LLAMA:
        default:
            enc_rc = semext_llama_encode(text, text_len, &raw_emb, &raw_dims);
            break;
    }
    if (enc_rc != 0 || !raw_emb) {
        free(raw_emb);
        sqlite3_result_error(ctx, "embedding encode failed", -1);
        return;
    }

    int out_dims = target_dims > 0 ? target_dims : raw_dims;

    // Build the f32 working copy with optional re-normalization skip,
    // dim truncation/zero-padding.
    float *normed = (float *)malloc(sizeof(float) * (size_t)out_dims);
    if (!normed) { free(raw_emb); sqlite3_result_error_nomem(ctx); return; }

    if (skip_renorm) {
        for (int i = 0; i < out_dims; i++)
            normed[i] = (i < raw_dims) ? raw_emb[i] : 0.0f;
    } else {
        // Re-normalize after any dim truncation (truncating a unit vector
        // doesn't leave it unit-norm).
        for (int i = 0; i < out_dims; i++)
            normed[i] = (i < raw_dims) ? raw_emb[i] : 0.0f;
        if (out_dims != raw_dims) {
            double norm = 0.0;
            for (int i = 0; i < out_dims; i++) norm += (double)normed[i] * (double)normed[i];
            norm = sqrt(norm);
            if (norm > 1e-12) {
                float inv = (float)(1.0 / norm);
                for (int i = 0; i < out_dims; i++) normed[i] *= inv;
            }
        }
    }
    free(raw_emb);

    // Encode into the on-disk blob (offset header + dtype-formatted
    // payload) via c_lib's EmbeddingCodec, through the C bridge.
    int total = 0;
    uint8_t *blob = diskerror_embedding_encode((int)vtype, blob_offset,
                                               normed, out_dims, &total);
    free(normed);
    if (!blob) { sqlite3_result_error_nomem(ctx); return; }

    sqlite3_result_blob(ctx, blob, total, SQLITE_TRANSIENT);
    free(blob);
}

/* ------------------------------------------------------------------ */
int semext_register_embed(sqlite3 *db) {
    ensure_config_table(db);
    sqlite3_create_function(db, "EMBED", 1, SQLITE_UTF8, NULL, embed_func, NULL, NULL);
    sqlite3_create_function(db, "SEMEXT_SET", 2, SQLITE_UTF8, NULL, semext_set_func, NULL, NULL);
    sqlite3_create_function(db, "SEMEXT_GET", 1, SQLITE_UTF8, NULL, semext_get_func, NULL, NULL);
    // Cleanup is registered lazily in ensure_model_loaded(), after the
    // first model load — see the comment there for why atexit ORDER
    // matters here (must run before ggml's own Metal-backend cleanup).
    return SQLITE_OK;
}
