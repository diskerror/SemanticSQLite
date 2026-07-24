// main.c — sqlite-ext: the REAL SQLite CLI shell (unmodified upstream
// vendor/sqlite/shell.c + sqlite3.c). Our two custom functions
// (DMPHON, EMBEDDING_SIM, EMBEDDING_DIST) are wired in via shell.c's own
// documented SQLITE_SHELL_EXTFUNCS extension point — see
// include/shell_extfuncs_hook.h and CMakeLists.txt for how that's plumbed
// through the build (-DSQLITE_SHELL_EXTFUNCS=SEMEXT -include hook.h).
//
// shell.c's own header comment documents `#define main sqlite3_shell` as
// the supported way for "other projects that use shell.c as a subroutine"
// (see vendor/sqlite/shell.c, ~line 38487). CMakeLists.txt compiles
// shell.c with -Dmain=sqlite3_shell so its main() becomes this callable
// subroutine. No upstream file is touched.

// The shell's real main(), renamed via -Dmain=sqlite3_shell at compile time.
extern int sqlite3_shell(int argc, char **argv);

int main(int argc, char **argv) {
    return sqlite3_shell(argc, argv);
}
