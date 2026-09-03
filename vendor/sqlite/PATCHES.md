# Carried patches to vendored SQLite

`shell.c`, `sqlite3.c`, and `sqlite3.h` are generated artifacts vendored from
the canonical `sqlite/sqlite` GitHub mirror (see `UPSTREAM_COMMIT.txt` for the
pinned commit + version). They are regenerated with
`make sqlite3.c shell.c sqlite3.h` and re-copied here.

**Any local edit to these files must be listed below and re-applied after every
re-vendor** — otherwise the change silently disappears on the next refresh.

SQLite's canonical source is Fossil, not Git, and the project does not accept
external pull requests, so these changes are **permanent carried patches**:
there is no upstream to merge them into. Re-apply each one after every
re-vendor.

---

## 1. Compiler-style error location prefix (`loc:line:`)

- **File:** `shell.c`
- **Function:** `runOneSqlLine()`
- **Status:** permanent carried patch — SQLite takes no external PRs, so this
  is re-applied on every re-vendor rather than upstreamed.
- **Marker:** search for `SEMQLITE PATCH` in `shell.c`.

**Why:** upstream prints SQL errors as `Error near line 13:` (stdin/heredoc,
no source shown) or `Error near line 13 of /full/absolute/path.sql:` (full
path). A bare line number is useless when sourcing multi-file SQL setups, and
the full path is noise.

**Change:** replace the `"%s near line %d[ of %s]:"` prefix block with a
compiler-style `loc:line:` prefix where `loc` is:
- `stdin`   — for `<stdin>`/heredoc/piped input (was: no source shown)
- `cmdline` — for `-cmd` / command-line SQL
- the **basename** of the file (last path component) — for `.read file.sql`
  (was: full absolute path)

So `Error near line 13:` becomes `stdin:13:`, and
`Error near line 13 of /Users/x/setup.sql:` becomes `setup.sql:13:`.

**Note:** this drops the leading error-type word (`Error` / `Parse error` /
`Runtime error`) from these two branches, per the chosen minimal
`loc:line:` format. The interactive-REPL branch (no filename, tty) is
unchanged and still prints `Error:`. If the type distinction is wanted back,
change the prefix to `"%s:%d: %s:"` with `zErrorType` as the trailing arg.

**To re-apply after a re-vendor:** find the `if( zFilename || !stdin_is_interactive )`
block in `runOneSqlLine()` and swap the three upstream `sqlite3_str_appendf`
location branches for the single `loc:line:` form (see the `SEMQLITE PATCH`
comment for the exact code).
