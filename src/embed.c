// embed.c — EMBED(text) via a GGUF model loaded through libllama, plus the
// semext_config table + SEMEXT_SET/SEMEXT_GET that stand in for the PRAGMAs
// SQLite extensions can't actually define (PRAGMA syntax is hardcoded into
// the core parser — not an extension point like functions/vtabs are).
//
// Config keys consulted by EMBED():
//   embedding_model        — path to a GGUF embedding model (required)
//   embedding_dims         — target output dimension, 1..4096 (optional;
//                             omit/0 to use the model's native n_embd).
//                             Larger than native = zero-padded; smaller =
//                             truncated (Matryoshka-style; not re-trained,
//                             just a slice — fine for experimentation, not
//                             claimed to preserve quality at all sizes).
//   embedding_vector_type  — "f16" (default) or "f32" storage in the BLOB
//
// The model is loaded lazily on first EMBED() call and cached for the
// process lifetime, keyed by path — calling SEMEXT_SET('embedding_model', X)
// with a different path swaps the cached model on the next EMBED() call.

#include "embed.h"

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

void semext_embed_shutdown(void) {
    unload_cached_model();
    if (g_backend_initialized) {
        llama_backend_free();
        g_backend_initialized = 0;
    }
}

// Ensures g_cache holds a loaded model for `path`. Returns 0 on success,
// -1 on failure (bad path, load error) with *errmsg set to a static or
// malloc'd message (malloc'd messages are the caller's to free — check
// *errmsg_owned).
static int ensure_model_loaded(const char *path, const char **errmsg) {
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

    // Register our cleanup AFTER the model/backend load, not at extension
    // registration time. atexit() runs handlers LIFO; ggml's Metal backend
    // registers its own static-object cleanup lazily, the first time a
    // Metal device is touched (i.e. during the llama_model_load_from_file
    // call above). Registering ours here — strictly after that point —
    // guarantees ours is later in the LIFO chain, so it runs FIRST at exit
    // and frees all llama/ggml buffers before ggml's own static destructor
    // asserts that everything was already freed. Registering it earlier
    // (e.g. at semext_register_embed() time, before any model ever loads)
    // put ours ahead of ggml's in the chain, so ggml's ran first and hit
    // "GGML_ASSERT([rsets->data count] == 0)" on process exit — confirmed
    // by reproducing and fixing this exact ordering bug.
    static int atexit_registered = 0;
    if (!atexit_registered) {
        atexit(semext_embed_shutdown);
        atexit_registered = 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* f32 -> f16 encode (portable, mirrors ext_functions.c's decode)      */
/* ------------------------------------------------------------------ */
static uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;

    if (exp <= 0) {
        return (uint16_t)sign;  // underflow to zero (signed)
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00u);  // overflow to inf
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
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

    char *model_path = config_get(db, "embedding_model");
    char *dims_str    = config_get(db, "embedding_dims");
    char *vtype_str   = config_get(db, "embedding_vector_type");

    const char *errmsg = NULL;
    if (ensure_model_loaded(model_path, &errmsg) != 0) {
        sqlite3_result_error(ctx, errmsg, -1);
        free(model_path); free(dims_str); free(vtype_str);
        return;
    }
    free(model_path);

    int target_dims = 0;  // 0 = use native
    if (dims_str && dims_str[0]) {
        target_dims = atoi(dims_str);
        if (target_dims < 0) target_dims = 0;
        if (target_dims > 4096) target_dims = 4096;
    }
    free(dims_str);

    int use_f32 = (vtype_str && strcmp(vtype_str, "f32") == 0);
    free(vtype_str);

    const struct llama_vocab *vocab = llama_model_get_vocab(g_cache.model);
    int n_embd = g_cache.n_embd;

    // Tokenize (add BOS/special as the model's tokenizer config dictates).
    int n_tokens_max = text_len + 8;
    llama_token *tokens = (llama_token *)malloc(sizeof(llama_token) * (size_t)n_tokens_max);
    if (!tokens) { sqlite3_result_error_nomem(ctx); return; }

    int n_tokens = llama_tokenize(vocab, text, text_len, tokens, n_tokens_max, true, true);
    if (n_tokens < 0) {
        n_tokens_max = -n_tokens + 8;
        llama_token *retry = (llama_token *)realloc(tokens, sizeof(llama_token) * (size_t)n_tokens_max);
        if (!retry) { free(tokens); sqlite3_result_error_nomem(ctx); return; }
        tokens = retry;
        n_tokens = llama_tokenize(vocab, text, text_len, tokens, n_tokens_max, true, true);
    }
    if (n_tokens <= 0) {
        free(tokens);
        sqlite3_result_error(ctx, "tokenization failed or produced no tokens", -1);
        return;
    }

    llama_memory_clear(llama_get_memory(g_cache.ctx), true);

    struct llama_batch batch = llama_batch_get_one(tokens, n_tokens);
    int rc = llama_decode(g_cache.ctx, batch);
    free(tokens);
    if (rc != 0) {
        sqlite3_result_error(ctx, "llama_decode failed", -1);
        return;
    }

    const float *embd = llama_get_embeddings_seq(g_cache.ctx, 0);
    if (!embd) {
        // MEAN pooling should populate seq embeddings; fall back to ith(0)
        // for models/configs where pooling didn't apply as expected.
        embd = llama_get_embeddings_ith(g_cache.ctx, 0);
    }
    if (!embd) {
        sqlite3_result_error(ctx, "failed to retrieve embeddings from context", -1);
        return;
    }

    // L2-normalize (matches Ragger's convention; nomic-bert etc. are not
    // pre-normalized by mean pooling alone).
    double norm = 0.0;
    for (int i = 0; i < n_embd; i++) norm += (double)embd[i] * (double)embd[i];
    norm = sqrt(norm);
    float inv_norm = (norm > 1e-12) ? (float)(1.0 / norm) : 0.0f;

    int out_dims = target_dims > 0 ? target_dims : n_embd;

    if (use_f32) {
        float *out = (float *)malloc(sizeof(float) * (size_t)out_dims);
        if (!out) { sqlite3_result_error_nomem(ctx); return; }
        for (int i = 0; i < out_dims; i++) {
            out[i] = (i < n_embd) ? embd[i] * inv_norm : 0.0f;
        }
        sqlite3_result_blob(ctx, out, (int)(sizeof(float) * (size_t)out_dims), SQLITE_TRANSIENT);
        free(out);
    } else {
        uint16_t *out = (uint16_t *)malloc(sizeof(uint16_t) * (size_t)out_dims);
        if (!out) { sqlite3_result_error_nomem(ctx); return; }
        for (int i = 0; i < out_dims; i++) {
            out[i] = (i < n_embd) ? f32_to_f16(embd[i] * inv_norm) : 0;
        }
        sqlite3_result_blob(ctx, out, (int)(sizeof(uint16_t) * (size_t)out_dims), SQLITE_TRANSIENT);
        free(out);
    }
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
