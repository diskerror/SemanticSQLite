// shell_extfuncs_hook.h — wires our custom functions into the REAL SQLite
// CLI shell via its own documented extension point (see vendor/sqlite/
// shell.c, search "SQLITE_SHELL_EXTFUNCS"): a two-macro protocol,
// <PREFIX>_INIT(db) and <PREFIX>_EXPOSE(db, pzErrMsg), invoked by shell.c's
// open_db() every time it opens a database connection (startup, .open,
// ATTACH). This fires safely *after* the shell's own sqlite3_config() calls,
// unlike sqlite3_auto_extension() registered from a wrapper main() (which
// forces early library init and trips shell.c's verify_uninitialized()
// self-check with spurious "attempt to configure SQLite after
// initialization" warnings).
//
// Wired in via CMake: -DSQLITE_SHELL_EXTFUNCS=SEMEXT -include this header,
// when compiling vendor/sqlite/shell.c. No upstream file is modified.
#ifndef SEMEXT_SHELL_EXTFUNCS_HOOK_H
#define SEMEXT_SHELL_EXTFUNCS_HOOK_H

#include <sqlite3.h>

/* Implemented in src/ext_functions.c — registers DMPHON(), EMBEDDING_SIM(),
 * EMBEDDING_DIST() on the given connection. */
extern int semext_register(sqlite3 *db, char **pzErrMsg, const void *pApi);

/* Implemented in src/embed.c — registers EMBED(), SEMQLITE_SET(), SEMQLITE_GET()
 * and ensures the semext_config table exists on the given connection. */
extern int semext_register_embed(sqlite3 *db);

static int semext_init_all(sqlite3 *db) {
    int rc = semext_register(db, 0, 0);
    if (rc != SQLITE_OK) return rc;
    return semext_register_embed(db);
}

#define SEMEXT_INIT(db)         semext_init_all((db))
#define SEMEXT_EXPOSE(db, msg)  ((void)0)

#endif /* SEMEXT_SHELL_EXTFUNCS_HOOK_H */
