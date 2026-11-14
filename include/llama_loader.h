/**
 * @file llama_loader.h
 * @brief Lazy, checked loading of libllama (llama.cpp).
 *
 * semqlite does NOT link libllama at build time. A hard link would make the
 * dynamic linker abort the whole process at start-up whenever the installed
 * llama.cpp changes its major version (e.g. a MacPorts/Homebrew upgrade turns
 * libllama.0 into libllama.1), taking every non-EMBED() SQL feature down with
 * it and giving the user no message. Instead the library is opened on the
 * first EMBED() that needs it, and every failure becomes a normal SQL error
 * that says what was expected, what was found, and what to do.
 *
 * The library opened is the exact soname the build was made against (a
 * different major version has a different struct layout, and llama.cpp has no
 * stable ABI, so "newest available" would be unsafe). `SEMQLITE_LLAMA_LIB`
 * (environment) overrides the path, for tests and unusual installs.
 */
#ifndef SEMQLITE_LLAMA_LOADER_H
#define SEMQLITE_LLAMA_LOADER_H

#include <stddef.h>
#include <llama.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The llama.cpp entry points semqlite uses, resolved with dlsym().
 *
 * Member types are taken from llama.h, so a prototype change in a newer
 * header is a compile error here rather than a runtime surprise.
 */
typedef struct semqlite_llama_api {
    __typeof__(&llama_backend_init)           backend_init;
    __typeof__(&llama_backend_free)           backend_free;
    __typeof__(&llama_model_default_params)   model_default_params;
    __typeof__(&llama_model_load_from_file)   model_load_from_file;
    __typeof__(&llama_model_free)             model_free;
    __typeof__(&llama_model_n_embd)           model_n_embd;
    __typeof__(&llama_model_get_vocab)        model_get_vocab;
    __typeof__(&llama_context_default_params) context_default_params;
    __typeof__(&llama_init_from_model)        init_from_model;
    __typeof__(&llama_free)                   free_context;
    __typeof__(&llama_tokenize)               tokenize;
    __typeof__(&llama_get_memory)             get_memory;
    __typeof__(&llama_memory_clear)           memory_clear;
    __typeof__(&llama_batch_get_one)          batch_get_one;
    __typeof__(&llama_decode)                 decode;
    __typeof__(&llama_get_embeddings_seq)     get_embeddings_seq;
    __typeof__(&llama_get_embeddings_ith)     get_embeddings_ith;
} semqlite_llama_api;

/**
 * @brief Open libllama (once) and resolve every entry point.
 *
 * @param api    Receives a pointer to the resolved table on success.
 * @param errmsg Receives a static, human-readable message on failure (valid
 *               until the next call).
 * @return 0 on success, -1 on failure. Failure is not cached: a later call
 *         tries again, so installing the right library needs no restart.
 */
int semqlite_llama_bind(const semqlite_llama_api **api, const char **errmsg);

/**
 * @brief Cheap availability check (no dlopen): does the library file exist?
 *
 * Used for the start-up warning and `-version`.
 *
 * @param buf Receives a description (OK line or the problem), always set.
 * @param n   Size of buf.
 * @return 0 if the expected library file exists, -1 otherwise.
 */
int semqlite_llama_probe(char *buf, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* SEMQLITE_LLAMA_LOADER_H */
