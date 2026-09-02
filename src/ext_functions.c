// ext_functions.c — custom SQL functions for sqlite-ext, registered via
// sqlite3_auto_extension() so they're available on every connection the
// real SQLite shell opens (see src/main.c).
//
//   DMPHON(text [, mode])        — Double Metaphone phonetic encoding
//   EMBEDDING_SIM(blob, blob)    — cosine similarity  (1.0 = identical direction)
//   EMBEDDING_DIST(blob, blob)   — cosine distance     (0.0 = identical direction)
//
// Embedding blobs are decoded using c_lib's EmbeddingCodec (via the
// EmbeddingCodecCapi C bridge, since this file is plain C) and the
// semext_config settings:
//   embedding_vector_type — "f16" (default), "f32", "bf16", or "int8"
//   embedding_offset      — byte offset into the blob where the vector payload
//                            starts (default 0; set to skip past any header —
//                            derive it by inspecting your DB's blobs; there
//                            is no universal default other than 0)
//
// Dimension count is NOT read from settings — it's inferred per-blob from
// its size (EmbeddingCodec's decode_infer_dims), since SemanticSQLite has
// no per-DB "expected dims" the way Ragger's settings table does.

#include "DoubleMetaphoneCapi.h"
#include "EmbeddingCodecCapi.h"

#include <sqlite3.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/* Config helpers (same semext_config table that embed.c manages)       */
/* ------------------------------------------------------------------ */
static char *ext_config_get(sqlite3 *db, const char *key) {
    sqlite3_stmt *stmt = NULL;
    char *result = NULL;
    if (sqlite3_prepare_v2(db,
            "SELECT value FROM semext_config WHERE key = ?",
            -1, &stmt, NULL) != SQLITE_OK) {
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

/* ------------------------------------------------------------------ */
/* Vector type identifiers (stable values, match Ragger convention and */
/* Diskerror::EmbeddingCodec::VectorType / diskerror_vector_type)       */
/* ------------------------------------------------------------------ */
enum vec_type { VT_F32 = 0, VT_F16 = 1, VT_BF16 = 2, VT_INT8 = 3 };

// Case-insensitive parse; returns VT_F16 for unrecognized strings.
static enum vec_type parse_vec_type(const char *s) {
    if (!s || s[0] == '\0') return VT_F16;
    // lowercase first char for fast dispatch
    char c0 = (s[0] >= 'A' && s[0] <= 'Z') ? (char)(s[0] - 'A' + 'a') : s[0];
    if (c0 == 'f') {
        if (s[1] == '3' && s[2] == '2') return VT_F32;
        if (s[1] == '1' && s[2] == '6') return VT_F16;
        return VT_F16;
    }
    if (c0 == 'b' || c0 == 'B') return VT_BF16;  // bf16 / bfloat16
    if (c0 == 'i' || c0 == 'q') return VT_INT8;  // int8 / i8 / q8
    return VT_F16;
}

// Read vtype + offset from semext_config (caches nothing — called per
// EMBEDDING_SIM/DIST invocation, but config reads are fast for a CLI tool).
static void read_embed_config(sqlite3 *db, enum vec_type *vt, int *offset) {
    *vt = VT_F16;
    *offset = 0;
    char *vtype_str  = ext_config_get(db, "embedding_vector_type");
    char *offset_str = ext_config_get(db, "embedding_offset");
    if (vtype_str)  { *vt = parse_vec_type(vtype_str); free(vtype_str); }
    if (offset_str) { *offset = atoi(offset_str); free(offset_str); }
}

// Shared cosine-similarity computation. Returns 0 on success, -1 on error
// (NULL/non-blob input, dimension mismatch, or zero vector).
static int cosine_similarity(sqlite3 *db,
                             sqlite3_value *v1, sqlite3_value *v2,
                             double *out) {
    if (sqlite3_value_type(v1) != SQLITE_BLOB ||
        sqlite3_value_type(v2) != SQLITE_BLOB) {
        return -1;
    }
    enum vec_type vt;
    int offset;
    read_embed_config(db, &vt, &offset);

    int d1 = 0, d2 = 0;
    float *a = diskerror_embedding_decode((int)vt, offset,
                           sqlite3_value_blob(v1), sqlite3_value_bytes(v1),
                           /*min_dims=*/16, &d1);
    float *b = diskerror_embedding_decode((int)vt, offset,
                           sqlite3_value_blob(v2), sqlite3_value_bytes(v2),
                           /*min_dims=*/16, &d2);
    if (!a || !b || d1 != d2 || d1 == 0) {
        diskerror_embedding_free(a); diskerror_embedding_free(b);
        return -1;
    }
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < d1; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    diskerror_embedding_free(a); diskerror_embedding_free(b);
    na = sqrt(na);
    nb = sqrt(nb);
    if (na == 0.0 || nb == 0.0) { *out = 0.0; return 0; }
    *out = dot / (na * nb);
    return 0;
}

/* ------------------------------------------------------------------ */
/* DMPHON(text [, mode])                                                */
/*   mode 0 (default): all words, both codes, space-separated          */
/*   mode 1: first code per word (primary — always present)            */
/*   mode 2: last code per word (alternate if it exists, else primary — */
/*           always present; only NULL/empty for non-alphabetic input) */
/* ------------------------------------------------------------------ */
static void dmphon_func(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 1 || sqlite3_value_type(argv[0]) == SQLITE_NULL) {
        sqlite3_result_null(ctx);
        return;
    }
    const char *text = (const char *)sqlite3_value_text(argv[0]);
    int mode = 0;
    if (argc >= 2 && sqlite3_value_type(argv[1]) != SQLITE_NULL) {
        mode = sqlite3_value_int(argv[1]);
    }

    char *result = diskerror_phonize_mode(text, mode);
    if (!result) {
        sqlite3_result_text(ctx, "", 0, SQLITE_STATIC);
        return;
    }
    sqlite3_result_text(ctx, result, -1, SQLITE_TRANSIENT);
    diskerror_free(result);
}

/* ------------------------------------------------------------------ */
/* EMBEDDING_SIM(blob1, blob2) -> cosine similarity [-1, 1]             */
/* EMBEDDING_DIST(blob1, blob2) -> cosine distance  [0, 2]              */
/* ------------------------------------------------------------------ */
static void embedding_sim_func(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2) { sqlite3_result_null(ctx); return; }
    sqlite3 *db = sqlite3_context_db_handle(ctx);
    double sim;
    if (cosine_similarity(db, argv[0], argv[1], &sim) != 0) {
        sqlite3_result_null(ctx);
        return;
    }
    sqlite3_result_double(ctx, sim);
}

static void embedding_dist_func(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2) { sqlite3_result_null(ctx); return; }
    sqlite3 *db = sqlite3_context_db_handle(ctx);
    double sim;
    if (cosine_similarity(db, argv[0], argv[1], &sim) != 0) {
        sqlite3_result_null(ctx);
        return;
    }
    sqlite3_result_double(ctx, 1.0 - sim);
}

/* ------------------------------------------------------------------ */
int semext_register(sqlite3 *db, char **pzErrMsg, const void *pApi) {
    (void)pzErrMsg;
    (void)pApi;
    sqlite3_create_function(db, "DMPHON", 1, SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                            NULL, dmphon_func, NULL, NULL);
    sqlite3_create_function(db, "DMPHON", 2, SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                            NULL, dmphon_func, NULL, NULL);
    sqlite3_create_function(db, "EMBEDDING_SIM", 2, SQLITE_UTF8,
                            NULL, embedding_sim_func, NULL, NULL);
    sqlite3_create_function(db, "EMBEDDING_DIST", 2, SQLITE_UTF8,
                            NULL, embedding_dist_func, NULL, NULL);
    return SQLITE_OK;
}
