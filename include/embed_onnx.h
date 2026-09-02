// embed_onnx.h — ONNX Runtime embedding backend for SemanticSQLite.
// Provides the same encode interface as the llama backend but using
// ONNX Runtime + tokenizers-cpp (matching Ragger's pipeline exactly).
//
// C interface so embed.c can call it without being compiled as C++.
#ifndef SEMQLITE_EMBED_ONNX_H
#define SEMQLITE_EMBED_ONNX_H

#ifdef __cplusplus
extern "C" {
#endif

// Load an ONNX embedding model from `model_dir`. Looks for model.onnx
// directly in model_dir, then falls back to model_dir/onnx/model.onnx
// (HF ONNX-export subfolder convention) — and tokenizer.json in model_dir.
// Returns 0 on success, -1 on failure with *errmsg set.
// Caches the model; calling with the same path is a no-op.
int semqlite_onnx_load(const char *model_dir, const char **errmsg);

// Encode `text` into a float embedding vector. `out` receives a malloc'd
// float array; `out_dims` receives the element count. Caller frees `out`.
// Returns 0 on success, -1 on failure.
int semqlite_onnx_encode(const char *text, int text_len,
                       float **out, int *out_dims);

// Free any cached model/session. Call at process exit.
void semqlite_onnx_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* SEMQLITE_EMBED_ONNX_H */
