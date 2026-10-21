/**
 * @file embed_llama.h
 * @brief Forward-declares the llama embedding backend so embed.c can
 * dispatch to it alongside embed_onnx.h.
 *
 * The actual implementation lives inline in embed.c (the existing llama
 * code), wrapped behind these function signatures.
 */
#ifndef SEMQLITE_EMBED_LLAMA_H
#define SEMQLITE_EMBED_LLAMA_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Load a GGUF embedding model.
 *
 * @param model_path Path to the .gguf file.
 * @param errmsg     Set to an error message on failure.
 * @return 0 on success, -1 on failure.
 */
int semqlite_llama_load(const char *model_path, const char **errmsg);

/**
 * @brief Encode text into a float embedding. Caller frees *out.
 *
 * @param text     Input bytes.
 * @param text_len Length of text in bytes.
 * @param out      Receives a malloc'd float array.
 * @param out_dims Receives the element count.
 * @return 0 on success, -1 on failure.
 */
int semqlite_llama_encode(const char *text, int text_len,
                        float **out, int *out_dims);

/** @brief Free cached model. */
void semqlite_llama_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* SEMQLITE_EMBED_LLAMA_H */
