// main.c — sqlite-ext: the REAL SQLite CLI shell (unmodified upstream
// vendor/sqlite/shell.c + sqlite3.c). Our custom functions (DMPHON,
// EMBEDDING_SIM, EMBEDDING_DIST, EMBED, SEMEXT_SET/GET) are wired in via
// shell.c's own documented SQLITE_SHELL_EXTFUNCS extension point — see
// include/shell_extfuncs_hook.h and CMakeLists.txt for how that's plumbed
// through the build (-DSQLITE_SHELL_EXTFUNCS=SEMEXT -include hook.h).
//
// shell.c's own header comment documents `#define main sqlite3_shell` as
// the supported way for "other projects that use shell.c as a subroutine"
// (see vendor/sqlite/shell.c, ~line 38487). CMakeLists.txt compiles
// shell.c with -Dmain=sqlite3_shell so its main() becomes this callable
// subroutine. No upstream file is touched.
//
// -version / --version: shell.c's own "-version" flag prints only the
// vendored SQLite library version and exits (see shell.c's usage() /
// "-version" handling) — it has no way to know about semext's own
// version or which optional backends were compiled in. We intercept
// -version/--version HERE, before handing off to sqlite3_shell(), print
// SQLite's version plus semext's own build info, and exit — without
// touching shell.c.

#include <sqlite3.h>
#include <stdio.h>
#include <string.h>

#ifndef SEMEXT_VERSION
#define SEMEXT_VERSION "0.1.0"
#endif

// The shell's real main(), renamed via -Dmain=sqlite3_shell at compile time.
extern int sqlite3_shell(int argc, char **argv);

static void print_semext_version(void) {
    printf("sqlite-ext %s (SemanticSQLite)\n", SEMEXT_VERSION);
    printf("  SQLite %s %s\n", sqlite3_libversion(), sqlite3_sourceid());
    printf("  Custom functions: DMPHON, EMBEDDING_SIM, EMBEDDING_DIST, "
           "EMBED, SEMEXT_SET, SEMEXT_GET\n");
    printf("  Embedding backends:"
#ifdef SEMEXT_HAVE_ONNX
           " onnx (default),"
#endif
           " llama\n");
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-version") == 0 || strcmp(argv[i], "--version") == 0) {
            print_semext_version();
            return 0;
        }
    }
    return sqlite3_shell(argc, argv);
}
