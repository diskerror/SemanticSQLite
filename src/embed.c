/**
 * @file embed.c
 * @brief EMBED(text) via either an ONNX model (default, matching Ragger's
 * pipeline exactly) or a GGUF model via libllama. Backend is selected by the
 * `embedding_embedder` config key ("onnx" or "llama").
 *
 * Config keys consulted by EMBED():
 *   embedding_embedder     — "onnx" (default) or "llama"
 *   embedding_model        — path to the model (ONNX: directory containing
 *                             model.onnx + tokenizer.json; llama: path to a
 *                             single .gguf file)
 *   embedding_dims         — target output dimension, 1..4096 (optional;
 *                             omit/0 to use the model's native n_embd).
 *                             Larger than native = zero-padded; smaller =
 *                             truncated (Matryoshka-style; not re-trained,
 *                             just a slice — fine for experimentation, not
 *                             claimed to preserve quality at all sizes).
 *   embedding_vector_type  — "f16" (default), "f32", "bf16", or "int8"
 *   embedding_offset       — byte offset prepended to the blob (default 0;
 *                             the offset bytes are zero-filled, allowing the
 *                             caller to reserve space for a header that an
 *                             external tool writes separately)
 *   embedding_skip_renorm  — "1"/"true"/"yes" to skip L2 normalization
 *
 * The model is loaded lazily on first EMBED() call and cached for the
 * process lifetime, keyed by path — calling SEMQLITE_SET('embedding_model', X)
 * with a different path swaps the cached model on the next EMBED() call.
 */

#include "embed.h"
#include "embed_llama.h"
#ifdef SEMQLITE_HAVE_ONNX
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
#include <sys/stat.h>

/**
 * @brief semqlite_config — key/value store standing in for PRAGMAs.
 *
 * Live settings are kept in temp.semqlite_config (RAM / connection
 * scoped); opening a database NEVER writes to it. If the database file
 * already holds a main.semqlite_config (made by SEMQLITE_SAVE()), its
 * rows are loaded into the temp table when the connection is set up.
 * Every key has a default; a missing known key is created with its
 * default on first read.
 */
static const struct { const char *key; const char *def; } k_config_defaults[] = {
    { "embedding_embedder",     "onnx" },
    { "embedding_model",        ""     },   /* no sensible default: unset */
    { "embedding_dims",         "0"    },   /* 0 = model's native dim */
    { "embedding_vector_type",  "f16"  },
    { "embedding_offset",       "0"    },
    { "embedding_skip_renorm",  "0"    },
    { "embedding_api_key",      ""     },   /* only saved by SEMQLITE_SAVE(1) */
};
#define K_API_KEY "embedding_api_key"
#define K_NDEFAULTS (sizeof k_config_defaults / sizeof k_config_defaults[0])

static const char *config_default(const char *key) {
    for (size_t i = 0; i < K_NDEFAULTS; i++)
        if (strcmp(k_config_defaults[i].key, key) == 0) return k_config_defaults[i].def;
    return NULL;
}

static int ensure_config_table(sqlite3 *db) {
    return sqlite3_exec(db,
        "CREATE TEMP TABLE IF NOT EXISTS semqlite_config ("
        "  key TEXT PRIMARY KEY,"
        "  value TEXT"
        ")", NULL, NULL, NULL);
}

static void config_set(sqlite3 *db, const char *key, const char *value) {
    ensure_config_table(db);
    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(db,
            "INSERT INTO temp.semqlite_config (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            -1, &stmt, NULL) != SQLITE_OK) {
        return;
    }
    sqlite3_bind_text(stmt, 1, key, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, value, -1, SQLITE_STATIC);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

/**
 * @brief Reads a config value into a newly malloc'd string (caller frees).
 *
 * A known key that is missing is created with its default and that default
 * returned; an unknown missing key returns NULL.
 *
 * Parameters are documented with the declaration in embed.h.
 */
char *semqlite_config_get(sqlite3 *db, const char *key) {
    ensure_config_table(db);
    sqlite3_stmt *stmt = NULL;
    char *result = NULL;
    if (sqlite3_prepare_v2(db, "SELECT value FROM temp.semqlite_config WHERE key = ?", -1, &stmt, NULL) != SQLITE_OK) {
        return NULL;
    }
    sqlite3_bind_text(stmt, 1, key, -1, SQLITE_STATIC);
    int found = (sqlite3_step(stmt) == SQLITE_ROW);
    if (found) {
        const char *v = (const char *)sqlite3_column_text(stmt, 0);
        if (!v) v = "";
        result = strdup(v);
    }
    sqlite3_finalize(stmt);
    if (!found) {
        const char *def = config_default(key);
        if (def) { config_set(db, key, def); result = strdup(def); }
    }
    return result;
}
#define config_get semqlite_config_get

/**
 * @brief Does main contain a table named semqlite_config?
 * @param db Open connection.
 * @return 1 if it exists, 0 otherwise.
 */
static int main_has_config(sqlite3 *db) {
    sqlite3_stmt *stmt = NULL;
    int has = 0;
    if (sqlite3_prepare_v2(db,
            "SELECT 1 FROM main.sqlite_master WHERE type='table' AND name='semqlite_config'",
            -1, &stmt, NULL) == SQLITE_OK) {
        has = (sqlite3_step(stmt) == SQLITE_ROW);
    }
    sqlite3_finalize(stmt);
    return has;
}

/**
 * @brief Connection setup: create the RAM table and load saved settings.
 *
 * If the table is brand new and the file carries a saved
 * main.semqlite_config, load those rows as if SEMQLITE_SET had been called
 * for each. Never writes to main.
 *
 * @param db Open connection.
 */
static void config_load_from_file(sqlite3 *db) {
    sqlite3_stmt *stmt = NULL;
    int exists = 0;
    if (sqlite3_prepare_v2(db,
            "SELECT 1 FROM sqlite_temp_master WHERE type='table' AND name='semqlite_config'",
            -1, &stmt, NULL) == SQLITE_OK) {
        exists = (sqlite3_step(stmt) == SQLITE_ROW);
    }
    sqlite3_finalize(stmt);
    if (exists) return;              /* already set up on this connection */
    ensure_config_table(db);
    if (main_has_config(db)) {
        sqlite3_exec(db,
            "INSERT OR REPLACE INTO temp.semqlite_config (key, value) "
            "SELECT key, value FROM main.semqlite_config", NULL, NULL, NULL);
    }
}

/**
 * @brief SEMQLITE_SAVE([include_api_key]): persist current settings into
 * main.semqlite_config.
 *
 * Every known key is first materialised with its default if missing, so the
 * saved table is never empty. The API key is saved only when include_api_key
 * is true; otherwise it is never written and any previously saved copy is
 * removed from the file.
 *
 * @param ctx  SQLite function context.
 * @param argc 0 or 1.
 * @param argv Optional include_api_key flag.
 * @note The number of keys now in the saved table is returned via ctx.
 */
static void semqlite_save_func(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    sqlite3 *db = sqlite3_context_db_handle(ctx);
    int with_key = (argc >= 1 && sqlite3_value_type(argv[0]) != SQLITE_NULL &&
                    sqlite3_value_int(argv[0]) != 0);
    for (size_t i = 0; i < K_NDEFAULTS; i++)
        free(config_get(db, k_config_defaults[i].key));
    char *err = NULL;
    int rc = sqlite3_exec(db,
        "SAVEPOINT semqlite_save;"
        "CREATE TABLE IF NOT EXISTS main.semqlite_config (key TEXT PRIMARY KEY, value TEXT);",
        NULL, NULL, &err);
    if (rc == SQLITE_OK) {
        const char *copy = with_key
            ? "INSERT INTO main.semqlite_config (key, value) "
              "  SELECT key, value FROM temp.semqlite_config WHERE true "
              "  ON CONFLICT(key) DO UPDATE SET value = excluded.value;"
            : "INSERT INTO main.semqlite_config (key, value) "
              "  SELECT key, value FROM temp.semqlite_config WHERE key <> '" K_API_KEY "' "
              "  ON CONFLICT(key) DO UPDATE SET value = excluded.value;"
              "DELETE FROM main.semqlite_config WHERE key = '" K_API_KEY "';";
        rc = sqlite3_exec(db, copy, NULL, NULL, &err);
    }
    if (rc == SQLITE_OK) rc = sqlite3_exec(db, "RELEASE semqlite_save;", NULL, NULL, &err);
    if (rc != SQLITE_OK) {
        sqlite3_exec(db, "ROLLBACK TO semqlite_save; RELEASE semqlite_save;", NULL, NULL, NULL);
        sqlite3_result_error(ctx, err ? err : "SEMQLITE_SAVE failed", -1);
        sqlite3_free(err);
        return;
    }
    sqlite3_stmt *stmt = NULL;
    int n = 0;
    if (sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM main.semqlite_config", -1, &stmt, NULL) == SQLITE_OK
        && sqlite3_step(stmt) == SQLITE_ROW)
        n = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    sqlite3_result_int(ctx, n);
}

static void semqlite_set_func(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2 || sqlite3_value_type(argv[0]) == SQLITE_NULL) {
        sqlite3_result_error(ctx, "SEMQLITE_SET(key, value): key required", -1);
        return;
    }
    sqlite3 *db = sqlite3_context_db_handle(ctx);
    const char *key = (const char *)sqlite3_value_text(argv[0]);
    const char *val = (sqlite3_value_type(argv[1]) == SQLITE_NULL)
                     ? "" : (const char *)sqlite3_value_text(argv[1]);
    config_set(db, key, val);
    sqlite3_result_text(ctx, val, -1, SQLITE_TRANSIENT);
}

static void semqlite_get_func(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 1 || sqlite3_value_type(argv[0]) == SQLITE_NULL) {
        sqlite3_result_null(ctx);
        return;
    }
    sqlite3 *db = sqlite3_context_db_handle(ctx);
    const char *key = (const char *)sqlite3_value_text(argv[0]);
    char *val = config_get(db, key);
    if (!val) { sqlite3_result_null(ctx); return; }
    sqlite3_result_text(ctx, val, -1, SQLITE_TRANSIENT);
    free(val);
}

/**
 * @brief libllama model cache — one model loaded at a time, keyed by path.
 *
 * Not thread-safe; fine for a single-connection CLI shell.
 */
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

void semqlite_llama_shutdown(void) {
    unload_cached_model();
    if (g_backend_initialized) {
        llama_backend_free();
        g_backend_initialized = 0;
    }
}

static void result_cache_clear(void);   /* defined with the EMBED() result cache below */

void semqlite_embed_shutdown(void) {
    result_cache_clear();
    semqlite_llama_shutdown();
#ifdef SEMQLITE_HAVE_ONNX
    semqlite_onnx_shutdown();
#endif
}

int semqlite_llama_load(const char *path, const char **errmsg) {
    *errmsg = NULL;
    if (!path || path[0] == '\0') {
        *errmsg = "embedding_model not set — call SEMQLITE_SET('embedding_model', '/path/to/model.gguf') first";
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

    /* atexit cleanup is registered centrally in embed_func() after
     * any successful backend load — not here — so the LIFO ordering
     * works for both llama-only and ONNX-only usage. */
    return 0;
}

int semqlite_llama_encode(const char *text, int text_len,
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

/**
 * @brief Vector type identifiers.
 *
 * Must match ext_functions.c's enum and diskerror_vector_type in c_lib's
 * EmbeddingCodecCapi.h.
 */
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

/** @brief Embedder backend identifiers. */
enum embed_backend { EB_ONNX = 0, EB_LLAMA = 1 };

static enum embed_backend parse_backend(const char *s) {
    if (!s || s[0] == '\0') {
#ifdef SEMQLITE_HAVE_ONNX
        return EB_ONNX;
#else
        return EB_LLAMA;
#endif
    }
    char c0 = (s[0] >= 'A' && s[0] <= 'Z') ? (char)(s[0] - 'A' + 'a') : s[0];
    if (c0 == 'l' || c0 == 'g') return EB_LLAMA;  // llama / gguf
    return EB_ONNX;
}


/**
 * @brief Expand a leading "~/" or bare "~" to $HOME.
 *
 * On any miss (no leading tilde, $HOME unset, or allocation failure)
 * returns a plain strdup of the input, so the caller can always free() the
 * result uniformly. A literal "~" not followed by '/' (e.g. "~user") is left
 * untouched — we don't resolve other users' homes.
 *
 * @param path Path to expand; may be NULL.
 * @return Newly malloc'd string the caller must free, or NULL if path is NULL.
 */
static char *expand_tilde(const char *path) {
    if (!path) return NULL;
    if (path[0] != '~' || (path[1] != '/' && path[1] != '\0'))
        return strdup(path);
    const char *home = getenv("HOME");
    if (!home || home[0] == '\0')
        return strdup(path);
    const char *rest = path + 1;               /* the part after '~' */
    size_t need = strlen(home) + strlen(rest) + 1;
    char *out = (char *)malloc(need);
    if (!out) return strdup(path);
    snprintf(out, need, "%s%s", home, rest);
    return out;
}

static char *g_model_dir = NULL;

void semqlite_set_model_directory(const char *dir) {
    free(g_model_dir);
    g_model_dir = (dir && dir[0]) ? strdup(dir) : NULL;
}

static int path_exists(const char *p) { struct stat st; return stat(p, &st) == 0; }

static char *join_path(const char *a, const char *b, const char *suffix) {
    size_t n = strlen(a) + 1 + strlen(b) + strlen(suffix) + 1;
    char *out = (char *)malloc(n);
    if (out) snprintf(out, n, "%s%s%s%s", a, (a[0] && a[strlen(a)-1] == '/') ? "" : "/", b, suffix);
    return out;
}

/**
 * @brief Model id -> path resolution.
 *
 *   - "/..." or "~..."  literal path
 *   - otherwise         \<--model-directory\>/\<id\>, else \<dir of DB\>/models/\<id\>
 *
 * llama also tries \<id\>.gguf. Exact names only — no globbing.
 *
 * @param db       Open connection (used to find the main DB file's directory).
 * @param id       Configured model id or path.
 * @param is_llama Nonzero to also try \<id\>.gguf.
 * @param out      On success, the resolved path (malloc'd).
 * @param emsg     On failure, the error message (malloc'd).
 * @return 0 on success; -1 with *emsg set otherwise.
 */
static int resolve_model(sqlite3 *db, const char *id, int is_llama, char **out, char **emsg) {
    char *cands[3] = {0}; int nc = 0;
    char *base = NULL;
    if (id[0] == '/' || id[0] == '~') {
        cands[nc++] = expand_tilde(id);
    } else {
        if (g_model_dir) {
            base = expand_tilde(g_model_dir);
        } else {
            const char *dbf = sqlite3_db_filename(db, "main");
            const char *slash = (dbf && dbf[0]) ? strrchr(dbf, '/') : NULL;
            if (slash) {
                size_t dl = (size_t)(slash - dbf);
                char *dir = (char *)malloc(dl + 1);
                if (dir) { memcpy(dir, dbf, dl); dir[dl] = '\0'; base = join_path(dir[0] ? dir : "/", "models", ""); free(dir); }
            }
        }
        if (!base) {
            size_t n = strlen(id) + 200;
            *emsg = (char *)malloc(n);
            if (*emsg) snprintf(*emsg, n, "embedding_model '%s' is not a path and the database has no file; "
                "use an absolute path or --model-directory=DIR", id);
            return -1;
        }
        cands[nc++] = join_path(base, id, "");
        if (is_llama) cands[nc++] = join_path(base, id, ".gguf");
    }
    free(base);
    for (int i = 0; i < nc; i++) {
        if (cands[i] && path_exists(cands[i])) {
            *out = cands[i];
            for (int j = 0; j < nc; j++) if (j != i) free(cands[j]);
            return 0;
        }
    }
    size_t n = 64;
    for (int i = 0; i < nc; i++) n += (cands[i] ? strlen(cands[i]) : 0) + 2;
    *emsg = (char *)malloc(n);
    if (*emsg) {
        strcpy(*emsg, "embedding_model not found; tried: ");
        for (int i = 0; i < nc; i++) {
            if (!cands[i]) continue;
            if (i) strcat(*emsg, ", ");
            strcat(*emsg, cands[i]);
        }
    }
    for (int i = 0; i < nc; i++) free(cands[i]);
    return -1;
}

/**
 * @brief Single-entry EMBED() result cache.
 *
 * SQLite calls EMBED() once per row, so a query like
 *   ... WHERE EMBEDDING_SIM(e, EMBED('some query')) > 0.5
 * would otherwise recompute the same query vector for every row.
 * Remember the last successful call; a full key match returns a copy
 * of its blob without loading the model or running the encoder.
 *
 * The key is the input bytes (length + memcmp — deliberately NOT a
 * hash: a collision would silently return the wrong embedding) plus
 * every setting that can change the output: embedder, the RESOLVED
 * model path (so a different --model-directory or DB models/ dir is a
 * different key), dims, vector type, offset and skip-renorm. The API
 * key is not part of the key (it does not change the vector).
 * Settings can be changed by a plain UPDATE of temp.semqlite_config,
 * so invalidation compares the current values on every call rather
 * than hooking SEMQLITE_SET. Only one entry: alternating two inputs
 * is always a miss (documented limitation).
 * Not thread-safe; fine for a single-connection CLI shell.
 */
typedef struct {
    int       valid;
    char     *text;         /* exact input bytes (may contain NUL) */
    size_t    text_len;
    char     *embedder;
    char     *model;        /* resolved model path; "" when unset */
    char     *dims;
    char     *vtype;
    char     *offset;
    char     *skip_renorm;
    uint8_t  *blob;
    size_t    blob_len;
} embed_result_entry;

static embed_result_entry g_result = {0};

static void result_cache_clear(void) {
    free(g_result.text);
    free(g_result.embedder);
    free(g_result.model);
    free(g_result.dims);
    free(g_result.vtype);
    free(g_result.offset);
    free(g_result.skip_renorm);
    free(g_result.blob);
    memset(&g_result, 0, sizeof g_result);
}

/**
 * @brief Compare a cached key string against the current one.
 *
 * NULL (unset model) is treated as "". Exact strcmp: "onnx" vs "ONNX" is
 * simply a different key — a miss is harmless.
 *
 * @param cached  Stored key string (never NULL).
 * @param current Current key string; may be NULL.
 * @return 1 if equal, 0 otherwise.
 */
static int key_str_eq(const char *cached, const char *current) {
    return strcmp(cached, current ? current : "") == 0;
}

static int result_cache_matches(const char *text, int text_len,
                                const char *embedder, const char *model,
                                const char *dims, const char *vtype,
                                const char *offset, const char *skip_renorm) {
    if (!g_result.valid || !text) return 0;
    if ((size_t)text_len != g_result.text_len) return 0;
    if (g_result.text_len && memcmp(g_result.text, text, g_result.text_len) != 0) return 0;
    return key_str_eq(g_result.embedder, embedder)
        && key_str_eq(g_result.model, model)
        && key_str_eq(g_result.dims, dims)
        && key_str_eq(g_result.vtype, vtype)
        && key_str_eq(g_result.offset, offset)
        && key_str_eq(g_result.skip_renorm, skip_renorm);
}

/**
 * @brief Replace the cached entry with a fresh copy of key + blob.
 *
 * Call only after a fully successful encode. On allocation failure the
 * cache is left empty rather than half-filled; the caller's result is
 * unaffected.
 *
 * @param text,text_len                                 Input bytes.
 * @param embedder,model,dims,vtype,offset,skip_renorm  Key strings.
 * @param blob,blob_len                                 Result to remember.
 */
static void result_cache_store(const char *text, int text_len,
                               const char *embedder, const char *model,
                               const char *dims, const char *vtype,
                               const char *offset, const char *skip_renorm,
                               const uint8_t *blob, int blob_len) {
    result_cache_clear();
    if (!text || blob_len <= 0) return;
    g_result.text = (char *)malloc((size_t)text_len + 1);
    if (g_result.text) {
        memcpy(g_result.text, text, (size_t)text_len);
        g_result.text[text_len] = '\0';
    }
    g_result.text_len = (size_t)text_len;
    g_result.embedder    = strdup(embedder    ? embedder    : "");
    g_result.model       = strdup(model       ? model       : "");
    g_result.dims        = strdup(dims        ? dims        : "");
    g_result.vtype       = strdup(vtype       ? vtype       : "");
    g_result.offset      = strdup(offset      ? offset      : "");
    g_result.skip_renorm = strdup(skip_renorm ? skip_renorm : "");
    g_result.blob = (uint8_t *)malloc((size_t)blob_len);
    if (g_result.blob) {
        memcpy(g_result.blob, blob, (size_t)blob_len);
        g_result.blob_len = (size_t)blob_len;
    }
    g_result.valid = g_result.text && g_result.embedder && g_result.model &&
                     g_result.dims && g_result.vtype && g_result.offset &&
                     g_result.skip_renorm && g_result.blob;
    if (!g_result.valid) result_cache_clear();
}

/**
 * @brief EMBED(text) -> BLOB embedding, per semqlite_config settings.
 *
 * Returns NULL for NULL input (never cached). Otherwise reads the config,
 * resolves the model path, and returns the cached blob on a full key match;
 * on a miss it loads the model, encodes, and stores the result.
 *
 * @param ctx  SQLite function context.
 * @param argc 1.
 * @param argv The text to embed.
 * @note The blob, NULL, or an error is returned via ctx.
 */
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
    char *model_path_raw = config_get(db, "embedding_model");
    char *model_path   = (model_path_raw && model_path_raw[0]) ? expand_tilde(model_path_raw) : NULL;
    free(model_path_raw);
    char *dims_str     = config_get(db, "embedding_dims");
    char *vtype_str    = config_get(db, "embedding_vector_type");
    char *offset_str   = config_get(db, "embedding_offset");
    char *skip_renorm_str = config_get(db, "embedding_skip_renorm");

    enum embed_backend backend = parse_backend(embedder_str);

    /* Resolve the configured model id to a path and fail early, listing every
     * path tried, if it doesn't exist. The cache key uses the RESOLVED path,
     * so a different --model-directory (or a different DB whose models/ dir
     * resolves the same id elsewhere) is a different key. An error here is
     * simply an error: the cache is neither consulted nor changed. */
    if (model_path) {
        char *resolved = NULL, *emsg = NULL;
        if (resolve_model(db, model_path, backend == EB_LLAMA, &resolved, &emsg) != 0) {
            sqlite3_result_error(ctx, emsg ? emsg : "embedding_model not found", -1);
            free(emsg);
            goto done;
        }
        free(model_path);
        model_path = resolved;
    }

    /* Cache hit: return a COPY of the last call's blob (SQLite must never
     * own or free the cached bytes) without loading the model or encoding. */
    if (result_cache_matches(text, text_len, embedder_str, model_path,
                             dims_str, vtype_str, offset_str, skip_renorm_str)) {
        sqlite3_result_blob(ctx, g_result.blob, (int)g_result.blob_len, SQLITE_TRANSIENT);
        goto done;
    }

    // Load the model via the selected backend.
    const char *errmsg = NULL;
    int load_rc;
    switch (backend) {
#ifdef SEMQLITE_HAVE_ONNX
        case EB_ONNX:
            load_rc = semqlite_onnx_load(model_path, &errmsg);
            break;
#endif
        case EB_LLAMA:
        default:
            load_rc = semqlite_llama_load(model_path, &errmsg);
            break;
    }
    if (load_rc != 0) {
        if (!errmsg) errmsg = "failed to load embedding model";
        sqlite3_result_error(ctx, errmsg, -1);
        goto done;
    }

    /* Ensure cleanup runs at exit for whichever backend(s) were loaded.
     * Must be registered AFTER any llama model load (atexit LIFO ordering
     * vs ggml's Metal cleanup — see the comment in semqlite_llama_load).
     * For ONNX-only usage, the llama backend was never touched so there's
     * no ggml ordering constraint, but we still need to free the ONNX env. */
    {
        static int atexit_registered = 0;
        if (!atexit_registered) {
            atexit(semqlite_embed_shutdown);
            atexit_registered = 1;
        }
    }

    int target_dims = 0;
    if (dims_str && dims_str[0]) {
        target_dims = atoi(dims_str);
        if (target_dims < 0) target_dims = 0;
        if (target_dims > 4096) target_dims = 4096;
    }

    enum embed_vec_type vtype = parse_embed_vec_type(vtype_str);

    int blob_offset = 0;
    if (offset_str && offset_str[0]) {
        blob_offset = atoi(offset_str);
        if (blob_offset < 0) blob_offset = 0;
    }

    int skip_renorm = 0;
    if (skip_renorm_str) {
        skip_renorm = (skip_renorm_str[0] == '1' ||
                       skip_renorm_str[0] == 't' || skip_renorm_str[0] == 'T' ||
                       skip_renorm_str[0] == 'y' || skip_renorm_str[0] == 'Y');
    }

    /* Encode via the selected backend. The backend returns a normalized f32
     * vector (both backends L2-normalize internally). */
    float *raw_emb = NULL;
    int raw_dims = 0;
    int enc_rc;
    switch (backend) {
#ifdef SEMQLITE_HAVE_ONNX
        case EB_ONNX:
            enc_rc = semqlite_onnx_encode(text, text_len, &raw_emb, &raw_dims);
            break;
#endif
        case EB_LLAMA:
        default:
            enc_rc = semqlite_llama_encode(text, text_len, &raw_emb, &raw_dims);
            break;
    }
    if (enc_rc != 0 || !raw_emb) {
        free(raw_emb);
        sqlite3_result_error(ctx, "embedding encode failed", -1);
        goto done;
    }

    int out_dims = target_dims > 0 ? target_dims : raw_dims;

    /* Build the f32 working copy with optional re-normalization skip,
     * dim truncation/zero-padding. */
    float *normed = (float *)malloc(sizeof(float) * (size_t)out_dims);
    if (!normed) { free(raw_emb); sqlite3_result_error_nomem(ctx); goto done; }

    if (skip_renorm) {
        for (int i = 0; i < out_dims; i++)
            normed[i] = (i < raw_dims) ? raw_emb[i] : 0.0f;
    } else {
        /* Re-normalize after any dim truncation (truncating a unit vector
         * doesn't leave it unit-norm). */
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

    /* Encode into the on-disk blob (offset header + dtype-formatted
     * payload) via c_lib's EmbeddingCodec, through the C bridge. */
    int total = 0;
    uint8_t *blob = diskerror_embedding_encode((int)vtype, blob_offset,
                                               normed, out_dims, &total);
    free(normed);
    if (!blob) { sqlite3_result_error_nomem(ctx); goto done; }

    sqlite3_result_blob(ctx, blob, total, SQLITE_TRANSIENT);

    /* Cache only fully successful results; every error path above leaves
     * the previous entry untouched. */
    result_cache_store(text, text_len, embedder_str, model_path, dims_str,
                       vtype_str, offset_str, skip_renorm_str, blob, total);
    free(blob);

done:
    free(embedder_str);
    free(model_path);
    free(dims_str);
    free(vtype_str);
    free(offset_str);
    free(skip_renorm_str);
}

/**
 * @brief Register EMBED, SEMQLITE_SET/GET/SAVE on a connection and load any
 * saved settings from the file.
 *
 * Parameters are documented with the declaration in embed.h.
 */
int semqlite_register_embed(sqlite3 *db) {
    config_load_from_file(db);
    sqlite3_create_function(db, "SEMQLITE_SAVE", 0, SQLITE_UTF8, NULL, semqlite_save_func, NULL, NULL);
    sqlite3_create_function(db, "SEMQLITE_SAVE", 1, SQLITE_UTF8, NULL, semqlite_save_func, NULL, NULL);
    sqlite3_create_function(db, "EMBED", 1, SQLITE_UTF8, NULL, embed_func, NULL, NULL);
    sqlite3_create_function(db, "SEMQLITE_SET", 2, SQLITE_UTF8, NULL, semqlite_set_func, NULL, NULL);
    sqlite3_create_function(db, "SEMQLITE_GET", 1, SQLITE_UTF8, NULL, semqlite_get_func, NULL, NULL);
    /* Cleanup is registered lazily in ensure_model_loaded(), after the
     * first model load — see the comment there for why atexit ORDER
     * matters here (must run before ggml's own Metal-backend cleanup). */
    return SQLITE_OK;
}
