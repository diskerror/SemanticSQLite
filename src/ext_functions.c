// ext_functions.c — custom SQL functions for sqlite-ext, registered via
// sqlite3_auto_extension() so they're available on every connection the
// real SQLite shell opens (see src/main.c).
//
//   DMPHON(text [, mode])        — Double Metaphone phonetic encoding
//   EMBEDDING_SIM(blob, blob)    — cosine similarity  (1.0 = identical direction)
//   EMBEDDING_DIST(blob, blob)   — cosine distance     (0.0 = identical direction)
//
// Embedding blobs are self-describing: f16 when byte_count == dim*2,
// f32 when byte_count == dim*4 (disambiguated via common embedding
// dimensions — see decode_blob()). Both blobs must have equal dimension.

#include "double_metaphone_capi.h"

#include <sqlite3.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/* f16 -> f32 decode (portable, no compiler _Float16 dependency)       */
/* ------------------------------------------------------------------ */
static float f16_to_f32(uint16_t bits) {
    uint32_t sign = (uint32_t)(bits >> 15) & 0x1u;
    uint32_t exp  = (uint32_t)(bits >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)bits & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            exp = 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            f = (sign << 31) | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(result));
    return result;
}

// Decode an embedding blob (f16 or f32) into a malloc'd float array.
// *out_dims receives the element count; caller frees the returned pointer.
// Disambiguates f16 vs f32 by preferring whichever byte-count maps to a
// common embedding dimension (384/512/768/1024/1536/2048/4096); falls back
// to treating a %4==0 byte count as f32.
static float *decode_blob(const void *data, int bytes, int *out_dims) {
    *out_dims = 0;
    if (!data || bytes <= 0) return NULL;

    int dims_f16 = (bytes % 2 == 0) ? bytes / 2 : -1;
    int dims_f32 = (bytes % 4 == 0) ? bytes / 4 : -1;

    int common[] = {384, 512, 768, 1024, 1536, 2048, 3072, 4096};
    int n_common = (int)(sizeof(common) / sizeof(common[0]));
    int f16_common = 0, f32_common = 0;
    for (int i = 0; i < n_common; i++) {
        if (dims_f16 == common[i]) f16_common = 1;
        if (dims_f32 == common[i]) f32_common = 1;
    }

    int use_f16;
    if (f16_common && !f32_common) use_f16 = 1;
    else if (f32_common && !f16_common) use_f16 = 0;
    else if (dims_f32 >= 0) use_f16 = 0;   /* ambiguous/unknown: prefer f32 */
    else if (dims_f16 >= 0) use_f16 = 1;
    else return NULL;

    if (use_f16) {
        float *out = (float *)malloc(sizeof(float) * (size_t)dims_f16);
        if (!out) return NULL;
        const uint16_t *h = (const uint16_t *)data;
        for (int i = 0; i < dims_f16; i++) out[i] = f16_to_f32(h[i]);
        *out_dims = dims_f16;
        return out;
    } else {
        float *out = (float *)malloc(sizeof(float) * (size_t)dims_f32);
        if (!out) return NULL;
        memcpy(out, data, sizeof(float) * (size_t)dims_f32);
        *out_dims = dims_f32;
        return out;
    }
}

// Shared cosine-similarity computation. Returns 0 on success, -1 on error
// (NULL/non-blob input, dimension mismatch, or zero vector).
static int cosine_similarity(sqlite3_value *v1, sqlite3_value *v2, double *out) {
    if (sqlite3_value_type(v1) != SQLITE_BLOB ||
        sqlite3_value_type(v2) != SQLITE_BLOB) {
        return -1;
    }
    int d1 = 0, d2 = 0;
    float *a = decode_blob(sqlite3_value_blob(v1), sqlite3_value_bytes(v1), &d1);
    float *b = decode_blob(sqlite3_value_blob(v2), sqlite3_value_bytes(v2), &d2);
    if (!a || !b || d1 != d2 || d1 == 0) {
        free(a); free(b);
        return -1;
    }
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < d1; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    free(a); free(b);
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

    char *result = semext_phonize_mode(text, mode);
    if (!result) {
        sqlite3_result_text(ctx, "", 0, SQLITE_STATIC);
        return;
    }
    sqlite3_result_text(ctx, result, -1, SQLITE_TRANSIENT);
    semext_free(result);
}

/* ------------------------------------------------------------------ */
/* EMBEDDING_SIM(blob1, blob2) -> cosine similarity [-1, 1]             */
/* EMBEDDING_DIST(blob1, blob2) -> cosine distance  [0, 2]              */
/* ------------------------------------------------------------------ */
static void embedding_sim_func(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2) { sqlite3_result_null(ctx); return; }
    double sim;
    if (cosine_similarity(argv[0], argv[1], &sim) != 0) {
        sqlite3_result_null(ctx);
        return;
    }
    sqlite3_result_double(ctx, sim);
}

static void embedding_dist_func(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2) { sqlite3_result_null(ctx); return; }
    double sim;
    if (cosine_similarity(argv[0], argv[1], &sim) != 0) {
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
    sqlite3_create_function(db, "EMBEDDING_SIM", 2, SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                            NULL, embedding_sim_func, NULL, NULL);
    sqlite3_create_function(db, "EMBEDDING_DIST", 2, SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                            NULL, embedding_dist_func, NULL, NULL);
    return SQLITE_OK;
}
