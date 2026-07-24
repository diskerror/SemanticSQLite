# SemanticSQLite

`sqlite-ext` — the **real, unmodified SQLite CLI shell** (vendored amalgamation,
built straight from the official [sqlite/sqlite](https://github.com/sqlite/sqlite)
mirror, forked as [diskerror/sqlite](https://github.com/diskerror/sqlite)),
with two custom SQL functions baked in for experimenting on Ragger's semantic
memory database.

## Why "real shell" and not a mini-REPL

`vendor/sqlite/shell.c` is upstream's own `src/shell.c.in`, run through their
own `tool/mkshellc.tcl`, byte-for-byte. **Zero patches.** Two things make that
possible:

1. **`#define main sqlite3_shell`** — documented directly in shell.c's own
   header comment above `int main()`: *"other projects that use shell.c as a
   subroutine"* rename it this way. `src/main.c` is a two-line wrapper that
   calls `sqlite3_shell(argc, argv)`.
2. **`SQLITE_SHELL_EXTFUNCS`** — a documented extension hook shell.c already
   has for exactly this purpose: bolting custom SQL functions onto every
   connection the shell opens, without touching shell.c itself. See
   `include/shell_extfuncs_hook.h`.

Net effect: you get the full upstream CLI — `.mode`, `.import`, `.dump`,
`.schema`, readline history, everything — plus `DMPHON()` and
`EMBEDDING_SIM()`/`EMBEDDING_DIST()` available on every database you open.

## Custom Functions

### `DMPHON(text [, mode])`
Double Metaphone phonetic encoding (Lawrence Philips, *CUJ* June 2000 — public
algorithm, ported from Ragger's C++ implementation with zero deps).

- **mode 0** (default): all phonetic codes for all words, space-separated
  (both primary and alternate when they differ)
- **mode 1**: primary code(s) only
- **mode 2**: secondary code(s) only (NULL if no alternate exists)

```sql
SELECT DMPHON('Don''t panic');   -- 'TNT PNK'
SELECT DMPHON('Schmidt', 1);     -- 'XMT'
SELECT DMPHON('Schmidt', 2);     -- 'SMT'
```

### `EMBEDDING_SIM(blob1, blob2)` / `EMBEDDING_DIST(blob1, blob2)`
Cosine similarity / cosine distance between two embedding BLOBs.
Auto-detects **f16** vs **f32** storage (Ragger stores f16 by default) by
matching common embedding dimensions (384/512/768/1024/1536/2048/3072/4096).

- `EMBEDDING_SIM`  → `[-1, 1]`, 1.0 = identical direction
- `EMBEDDING_DIST` → `[0, 2]`, 0.0 = identical direction (`1 - cosine`)

```sql
-- Most semantically similar summaries to summary_id=2
SELECT s2.summary_id, EMBEDDING_SIM(s1.embedding, s2.embedding) AS sim
FROM summaries s1, summaries s2
WHERE s1.summary_id = 2 AND s2.summary_id != 2
ORDER BY sim DESC LIMIT 10;
```

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Produces `sqlite-ext` — drop-in `sqlite3` replacement.

## Usage

```bash
sqlite-ext ~/.ragger/memories.db              # interactive shell
sqlite-ext ~/.ragger/memories.db "SELECT ..."  # one-shot query
```

## Refreshing the vendored SQLite source

The amalgamation was generated from `diskerror/sqlite` (a fork of the
official mirror) at the commit recorded in
`vendor/sqlite/UPSTREAM_COMMIT.txt`:

```bash
cd /path/to/sqlite-fork && git pull
mkdir build-amal && cd build-amal
../configure --disable-tcl
make sqlite3.c shell.c sqlite3.h
cp sqlite3.c sqlite3.h shell.c /path/to/SemanticSQLite/vendor/sqlite/
```

## License

SQLite itself is public domain. `DMPHON`/Double Metaphone is a published,
widely-implemented public algorithm. This project's own code (the
`EMBEDDING_SIM`/`EMBEDDING_DIST` implementation, the shell-integration glue)
is released with no restrictions — do whatever you want with it.
