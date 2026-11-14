/**
 * @file llama_loader.c
 * @brief Lazy, checked loading of libllama; see llama_loader.h.
 */
#include "llama_loader.h"

#include <dirent.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* Set by CMake from the library the build was made against. */
#ifndef SEMQLITE_LLAMA_SONAME
#  ifdef __APPLE__
#    define SEMQLITE_LLAMA_SONAME "libllama.dylib"
#  else
#    define SEMQLITE_LLAMA_SONAME "libllama.so"
#  endif
#endif
#ifndef SEMQLITE_LLAMA_LIBDIR
#  define SEMQLITE_LLAMA_LIBDIR ""
#endif

static semqlite_llama_api g_api;
static void *g_handle = NULL;
static char g_err[1536];

/**
 * @brief Path to try first: $SEMQLITE_LLAMA_LIB, else \<build dir\>/\<soname\>.
 *
 * @param buf Receives the path.
 * @param n   Size of buf.
 * @return 1 if the path came from the environment override, else 0.
 */
static int primary_path(char *buf, size_t n) {
    const char *env = getenv("SEMQLITE_LLAMA_LIB");
    if (env && env[0]) { snprintf(buf, n, "%s", env); return 1; }
    if (SEMQLITE_LLAMA_LIBDIR[0])
        snprintf(buf, n, "%s/%s", SEMQLITE_LLAMA_LIBDIR, SEMQLITE_LLAMA_SONAME);
    else
        snprintf(buf, n, "%s", SEMQLITE_LLAMA_SONAME);
    return 0;
}

/**
 * @brief Append the names of libllama* files in the build's library dir.
 *
 * Tells the user what is installed instead (e.g. libllama.1.dylib when the
 * build wants libllama.0.dylib).
 *
 * @param out Buffer to append to.
 * @param n   Size of out.
 */
static void list_found(char *out, size_t n) {
    if (!SEMQLITE_LLAMA_LIBDIR[0]) return;
    DIR *d = opendir(SEMQLITE_LLAMA_LIBDIR);
    if (!d) return;
    size_t used = strlen(out);
    int count = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (strncmp(e->d_name, "libllama.", 9) != 0) continue;
        int w = snprintf(out + used, n - used, "%s%s", count ? ", " : "", e->d_name);
        if (w < 0 || (size_t)w >= n - used) break;
        used += (size_t)w;
        count++;
    }
    closedir(d);
    if (!count) snprintf(out + used, n - used, "(no libllama* files)");
}

/** @brief Fixed tail of every "cannot use libllama" message. */
#define REBUILD_HINT \
    "EMBED() with GGUF models is unavailable; everything else works. " \
    "Rebuild SemanticSQLite against the installed llama.cpp " \
    "(see README: Building) or set SEMQLITE_LLAMA_LIB."

int semqlite_llama_probe(char *buf, size_t n) {
    char path[1024];
    int from_env = primary_path(path, sizeof path);
    struct stat st;
    /* A bare soname (no build dir recorded) cannot be checked without
     * dlopen; report it as unchecked rather than guessing. */
    if (!from_env && !SEMQLITE_LLAMA_LIBDIR[0]) {
        snprintf(buf, n, "libllama: %s (not checked)", path);
        return 0;
    }
    if (stat(path, &st) == 0) {
        snprintf(buf, n, "libllama: %s", path);
        return 0;
    }
    char found[512] = "";
    list_found(found, sizeof found);
    snprintf(buf, n,
             "libllama: %s not found (installed: %s). This semqlite was "
             "built against a different llama.cpp. " REBUILD_HINT,
             path, found[0] ? found : "none");
    return -1;
}

int semqlite_llama_bind(const semqlite_llama_api **api, const char **errmsg) {
    *errmsg = NULL;
    if (g_handle) { *api = &g_api; return 0; }

    char path[1024];
    int from_env = primary_path(path, sizeof path);
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h && !from_env && SEMQLITE_LLAMA_LIBDIR[0])
        h = dlopen(SEMQLITE_LLAMA_SONAME, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        char found[512] = "";
        list_found(found, sizeof found);
        const char *why = dlerror();
        snprintf(g_err, sizeof g_err,
                 "cannot load %s (%s). Installed here: %s. This semqlite was "
                 "built against a different llama.cpp or a dependency of it "
                 "is missing. " REBUILD_HINT,
                 path, why ? why : "unknown error",
                 found[0] ? found : "n/a");
        *errmsg = g_err;
        return -1;
    }

    semqlite_llama_api t;
    memset(&t, 0, sizeof t);
#define BIND(field, sym)                                                     \
    do {                                                                     \
        *(void **)(&t.field) = dlsym(h, sym);                                \
        if (!t.field) {                                                      \
            snprintf(g_err, sizeof g_err,                                    \
                     "%s has no symbol '%s': it is a different llama.cpp "   \
                     "version than semqlite was built against. "             \
                     REBUILD_HINT, path, sym);                               \
            dlclose(h);                                                      \
            *errmsg = g_err;                                                 \
            return -1;                                                       \
        }                                                                    \
    } while (0)
    BIND(backend_init,          "llama_backend_init");
    BIND(backend_free,          "llama_backend_free");
    BIND(model_default_params,  "llama_model_default_params");
    BIND(model_load_from_file,  "llama_model_load_from_file");
    BIND(model_free,            "llama_model_free");
    BIND(model_n_embd,          "llama_model_n_embd");
    BIND(model_get_vocab,       "llama_model_get_vocab");
    BIND(context_default_params,"llama_context_default_params");
    BIND(init_from_model,       "llama_init_from_model");
    BIND(free_context,          "llama_free");
    BIND(tokenize,              "llama_tokenize");
    BIND(get_memory,            "llama_get_memory");
    BIND(memory_clear,          "llama_memory_clear");
    BIND(batch_get_one,         "llama_batch_get_one");
    BIND(decode,                "llama_decode");
    BIND(get_embeddings_seq,    "llama_get_embeddings_seq");
    BIND(get_embeddings_ith,    "llama_get_embeddings_ith");
#undef BIND

    g_api = t;
    g_handle = h;
    *api = &g_api;
    return 0;
}
