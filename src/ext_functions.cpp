// ext_functions.cpp — custom SQL functions for sqlite-ext.
//
//   DMPHON(text [, mode])       — Double Metaphone phonetic encoding
//   EMBEDDING_SCORE(blob, blob) — cosine similarity between two embedding blobs
//
// Embedding blobs are self-describing: f16 when byte_count == dim*2,
// f32 when byte_count == dim*4. Both blobs must have the same dimension.

#include "ext_functions.h"
#include "double_metaphone.h"

#include <Eigen/Dense>
#include <cstring>
#include <string>
#include <vector>

namespace semext {

// -----------------------------------------------------------------------
// f16 ↔ f32 helpers (Apple Silicon native _Float16)
// -----------------------------------------------------------------------
#if defined(__aarch64__) || defined(__arm64__)
static inline float f16_to_f32(uint16_t bits) {
    _Float16 h;
    std::memcpy(&h, &bits, sizeof(h));
    return static_cast<float>(h);
}
#else
// Software fallback for x86 — IEEE 754 half-precision decode
static inline float f16_to_f32(uint16_t bits) {
    uint32_t sign = (bits >> 15) & 0x1;
    uint32_t exp  = (bits >> 10) & 0x1F;
    uint32_t mant = bits & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign << 31; }
        else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}
#endif

// Decode an embedding blob (f16 or f32, auto-detected by size) into floats.
static std::vector<float> decode_blob(const void* data, int bytes) {
    if (!data || bytes == 0) return {};
    // Try f32 first (must be multiple of 4)
    if (bytes % 4 == 0 && bytes % 2 == 0) {
        // Ambiguous — could be either. Heuristic: if bytes/2 gives a common
        // embedding dim (384, 768, 1024, 1536, 4096) treat as f16.
        int dims_f16 = bytes / 2;
        int dims_f32 = bytes / 4;
        bool likely_f16 = (dims_f16 == 384 || dims_f16 == 768 || dims_f16 == 1024 ||
                           dims_f16 == 1536 || dims_f16 == 4096);
        bool likely_f32 = (dims_f32 == 384 || dims_f32 == 768 || dims_f32 == 1024 ||
                           dims_f32 == 1536 || dims_f32 == 4096);
        if (likely_f16 && !likely_f32) {
            // f16
            std::vector<float> out(dims_f16);
            auto* h = static_cast<const uint16_t*>(data);
            for (int i = 0; i < dims_f16; ++i) out[i] = f16_to_f32(h[i]);
            return out;
        }
        if (likely_f32) {
            std::vector<float> out(dims_f32);
            std::memcpy(out.data(), data, bytes);
            return out;
        }
    }
    // Fall through: try f32 if multiple of 4, else f16 if multiple of 2
    if (bytes % 4 == 0) {
        int dims = bytes / 4;
        std::vector<float> out(dims);
        std::memcpy(out.data(), data, bytes);
        return out;
    }
    if (bytes % 2 == 0) {
        int dims = bytes / 2;
        std::vector<float> out(dims);
        auto* h = static_cast<const uint16_t*>(data);
        for (int i = 0; i < dims; ++i) out[i] = f16_to_f32(h[i]);
        return out;
    }
    return {};
}

// -----------------------------------------------------------------------
// DMPHON(text [, mode])
//   mode 0 (default): both codes space-separated (primary [secondary])
//   mode 1: primary only
//   mode 2: secondary only (NULL if none)
// -----------------------------------------------------------------------
static void dmphon_func(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    if (argc < 1 || sqlite3_value_type(argv[0]) == SQLITE_NULL) {
        sqlite3_result_null(ctx);
        return;
    }

    const char* text = reinterpret_cast<const char*>(sqlite3_value_text(argv[0]));
    int mode = 0;
    if (argc >= 2 && sqlite3_value_type(argv[1]) != SQLITE_NULL) {
        mode = sqlite3_value_int(argv[1]);
    }

    if (mode == 0) {
        // Full phonize — all words, both codes
        std::string result = phonize(text);
        sqlite3_result_text(ctx, result.c_str(), static_cast<int>(result.size()),
                            SQLITE_TRANSIENT);
        return;
    }

    // Single-word mode for mode 1/2 doesn't make sense on multi-word input,
    // but we'll phonize word-by-word and pick the requested code per word.
    std::string result;
    std::string word;
    auto flush = [&](const std::string& w) {
        if (w.empty()) return;
        auto codes = double_metaphone(w);
        if (codes.empty()) return;
        std::string picked;
        if (mode == 1) {
            picked = codes[0];  // primary
        } else if (mode == 2) {
            picked = (codes.size() > 1) ? codes[1] : "";
        }
        if (picked.empty()) return;
        if (!result.empty()) result += ' ';
        result += picked;
    };

    for (unsigned char c : std::string_view(text)) {
        if (std::isalpha(c) || c == '\'') word += static_cast<char>(c);
        else { flush(word); word.clear(); }
    }
    flush(word);

    if (result.empty() && mode == 2) {
        sqlite3_result_null(ctx);
    } else {
        sqlite3_result_text(ctx, result.c_str(), static_cast<int>(result.size()),
                            SQLITE_TRANSIENT);
    }
}

// -----------------------------------------------------------------------
// EMBEDDING_SCORE(blob1, blob2)
//   Cosine similarity between two embedding blobs.
//   Returns NULL if either is NULL or dimensions mismatch.
// -----------------------------------------------------------------------
static void embedding_score_func(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    if (argc < 2 ||
        sqlite3_value_type(argv[0]) != SQLITE_BLOB ||
        sqlite3_value_type(argv[1]) != SQLITE_BLOB) {
        sqlite3_result_null(ctx);
        return;
    }

    auto v1 = decode_blob(sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]));
    auto v2 = decode_blob(sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]));

    if (v1.empty() || v2.empty() || v1.size() != v2.size()) {
        sqlite3_result_null(ctx);
        return;
    }

    int dims = static_cast<int>(v1.size());
    Eigen::Map<Eigen::VectorXf> a(v1.data(), dims);
    Eigen::Map<Eigen::VectorXf> b(v2.data(), dims);

    float norm_a = a.norm();
    float norm_b = b.norm();
    if (norm_a == 0.0f || norm_b == 0.0f) {
        sqlite3_result_double(ctx, 0.0);
        return;
    }

    float cosine = a.dot(b) / (norm_a * norm_b);
    sqlite3_result_double(ctx, static_cast<double>(cosine));
}

// -----------------------------------------------------------------------
void register_functions(sqlite3* db) {
    sqlite3_create_function(db, "DMPHON", 1, SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                            nullptr, dmphon_func, nullptr, nullptr);
    sqlite3_create_function(db, "DMPHON", 2, SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                            nullptr, dmphon_func, nullptr, nullptr);
    sqlite3_create_function(db, "EMBEDDING_SCORE", 2, SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                            nullptr, embedding_score_func, nullptr, nullptr);
}

} // namespace semext
