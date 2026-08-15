// embed_llama.h — forward-declares the llama embedding backend so embed.c
// can dispatch to it alongside embed_onnx.h. The actual implementation
// lives inline in embed.c (the existing llama code), wrapped behind these
// function signatures.
#ifndef SEMEXT_EMBED_LLAMA_H
#define SEMEXT_EMBED_LLAMA_H

#ifdef __cplusplus
extern "C" {
#endif

// Load a GGUF embedding model. Returns 0 on success, -1 on failure.
int semext_llama_load(const char *model_path, const char **errmsg);

// Encode text into a float embedding. Caller frees *out.
int semext_llama_encode(const char *text, int text_len,
                        float **out, int *out_dims);

// Free cached model.
void semext_llama_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* SEMEXT_EMBED_LLAMA_H */
