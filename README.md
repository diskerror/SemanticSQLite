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
`.schema`, readline history, everything — plus `DMPHON()`,
`EMBEDDING_SIM()`/`EMBEDDING_DIST()`, and `EMBED()` available on every
database you open.

## Custom Functions

### `DMPHON(text [, mode])`
Double Metaphone phonetic encoding (Lawrence Philips, *CUJ* June 2000 — public
algorithm, ported from Ragger's C++ implementation with zero deps).

- **mode 0** (default): all phonetic codes for all words, space-separated
  (both primary and alternate when they differ)
- **mode 1**: first code per word (primary — always present)
- **mode 2**: last code per word (alternate if one exists, else primary —
  always present, never NULL for alphabetic input)

```sql
SELECT DMPHON('Don''t panic');   -- 'TNT PNK'
SELECT DMPHON('Schmidt', 1);     -- 'XMT'
SELECT DMPHON('Schmidt', 2);     -- 'SMT'
SELECT DMPHON('hello', 2);       -- 'HL'  (only one code exists, so mode 2 == mode 1)
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

### `EMBED(text)`
Generates a sentence embedding via a GGUF model, loaded through **libllama**
(MacPorts `llama.cpp` package). Mean-pooled, L2-normalized, returned as a
BLOB — same shape as embeddings already stored in Ragger's DB.

SQLite extensions can't actually define new `PRAGMA`s (the syntax is
hardcoded into the core parser, not an extension point like functions/vtabs
are), so the closest equivalent — settings that persist in the database file
so you never re-type them — is a `semext_config` table plus two helper
functions:

```sql
SELECT SEMEXT_SET('embedding_model', '/path/to/nomic-embed-text-v1.5.Q4_K_M.gguf');
SELECT SEMEXT_SET('embedding_dims', '512');            -- optional, 1..4096; omit/0 = model's native dim
SELECT SEMEXT_SET('embedding_vector_type', 'f32');     -- optional, "f16" (default) or "f32"

SELECT SEMEXT_GET('embedding_model');   -- read back current setting

SELECT EMBED('some text to embed');     -- BLOB, per the settings above
```

Settings are set once per database and persist across `sqlite-ext` restarts
(they live in `semext_config`, auto-created on first use). The model itself
is loaded lazily on first `EMBED()` call and cached for the process
lifetime, keyed by path — calling `SEMEXT_SET('embedding_model', ...)` with
a different path swaps the cached model on the next call. 

`embedding_dims` smaller than the model's native output truncates
(Matryoshka-style slicing — not re-trained for it, just a slice, fine for
experimentation); larger zero-pads.

Any GGUF text-embedding model works (mean/CLS/last pooling all handled via
`llama_context_params.pooling_type = LLAMA_POOLING_TYPE_MEAN`). Verified
against `nomic-embed-text-v1.5` (768-dim, nomic-bert architecture).

```sql
-- Sanity check: related text scores higher than unrelated text
SELECT EMBEDDING_SIM(EMBED('cats are great pets'), EMBED('dogs are wonderful companions')); -- ~0.49
SELECT EMBEDDING_SIM(EMBED('cats are great pets'), EMBED('quantum chromodynamics'));        -- ~0.01
```

## Build

Requires `libllama` (for `EMBED()`):

```bash
sudo port install llama.cpp   # macOS/MacPorts — provides libllama.dylib + llama.h under /opt/local
```

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
widely-implemented public algorithm. `EMBED()` links libllama (MIT license).
This project's own code (the `EMBEDDING_SIM`/`EMBEDDING_DIST`/`EMBED`
implementations, the shell-integration glue) is released with no
restrictions — do whatever you want with it.
