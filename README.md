# SemanticSQLite

`semqlite` — the **real, unmodified SQLite CLI shell** (vendored amalgamation,
built straight from the official [sqlite/sqlite](https://github.com/sqlite/sqlite)
mirror, with six custom SQL functions baked in for experimenting on Ragger's semantic
memory database. It assumes that for the most part embeddings are stored as a blob of the binary 
vector. Other details of the vector encoding can be configured with a few commands.

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
`EMBEDDING_SIM()`/`EMBEDDING_DIST()`, `EMBED()`, and `STEM_PORTER()`/
`STEM_SNOWBALL()` available on every database you open.

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

### `STEM_PORTER(word)`
Stems a single English word using the classic 1980 Porter algorithm
(ported from Ragger's C++ implementation, zero deps). English only, no
configuration.

```sql
SELECT STEM_PORTER('running');   -- 'run'
SELECT STEM_PORTER('flies');     -- 'fli'
SELECT STEM_PORTER('RUNNING');   -- 'run'  (case-insensitive)
```

### `STEM_SNOWBALL(word [, language])`
Stems a single UTF-8 word using the [Snowball](https://snowballstem.org)
stemming algorithm family (vendored `libstemmer_c`). Unlike `STEM_PORTER`,
this supports many languages — one stemming algorithm per language, picked
by name at call time.

`language` accepts either a canonical long name (`english`, `french`,
`german`, `russian`, ...) or a short code (`en`, `fr`, `de`, `ru`, ...),
case-insensitively. The full list of supported languages: arabic (`ar`),
armenian (`hy`), basque (`eu`), catalan (`ca`), danish (`da`), dutch (`nl`),
english (`en`), finnish (`fi`), french (`fr`), german (`de`), greek (`el`),
hindi (`hi`), hungarian (`hu`), indonesian (`id`), irish (`ga`), italian
(`it`), lithuanian (`lt`), nepali (`ne`), norwegian (`no`), **porter**
(long name only — the original 1980 Porter algorithm as implemented by
Snowball itself, distinct from `STEM_PORTER()` above), portuguese (`pt`),
romanian (`ro`), russian (`ru`), serbian (`sr`), spanish (`es`), swedish
(`sv`), tamil (`ta`), turkish (`tr`), yiddish (`yi`).

If `language` is omitted, `STEM_SNOWBALL` uses the `snowball_language`
`semqlite_config` setting, which itself defaults to English (`en`) if
never set:

```sql
SELECT SEMQLITE_SET('snowball_language', 'en');  -- set the default (optional; 'en' is already the default)

SELECT STEM_SNOWBALL('running');            -- 'run'          (default English)
SELECT STEM_SNOWBALL('chevaux', 'french');  -- 'cheval'       (long name)
SELECT STEM_SNOWBALL('chevaux', 'fr');      -- 'cheval'       (short code, same result)
SELECT STEM_SNOWBALL('laufen', 'de');       -- 'lauf'
```

An unrecognised language name degrades gracefully to a lowercased
passthrough of the input rather than an error — check
`snowball_language_supported()`-style validation up front if you need to
detect that case.

**Note:** `STEM_SNOWBALL` with the 1-argument form is **not** marked
`SQLITE_DETERMINISTIC` because its result depends on the mutable
`snowball_language` setting; the 2-argument form (explicit language) is
deterministic. `STEM_PORTER` is always deterministic.

### `EMBEDDING_SIM(blob1, blob2)` / `EMBEDDING_DIST(blob1, blob2)`
Cosine similarity / cosine distance between two embedding BLOBs.
The storage dtype and byte layout are controlled by `semqlite_config` settings:

| key | values | default | purpose |
|-----|--------|---------|---------|
| `embedding_vector_type` | `f32`, `f16`, `bf16`, `int8` | `f16` | How the payload bytes are interpreted |
| `embedding_offset` | integer (bytes) | `0` | Bytes to skip at the start of each blob before the vector payload — set this to match any header your application prepends |

- `EMBEDDING_SIM`  → `[-1, 1]`, 1.0 = identical direction
- `EMBEDDING_DIST` → `[0, 2]`, 0.0 = identical direction (`1 - cosine`)

**int8 payloads:** N bytes of quantized data, optionally followed by a 2-byte
IEEE f16 dequantization scale (scale = max|x| / 127). If the scale suffix is
absent, a fallback of 1/127 is used (correct for unit-norm vectors).

**Note:** these functions are **not** marked `SQLITE_DETERMINISTIC` because
their behavior depends on the mutable `semqlite_config` settings. Same blob
inputs with a different `embedding_vector_type` produce different results.

```sql
-- Configure for Ragger's default: f16 blobs with 1-byte version prefix
SELECT SEMQLITE_SET('embedding_vector_type', 'f16');
SELECT SEMQLITE_SET('embedding_offset', '1');

-- Most semantically similar summaries to summary_id=2
SELECT s2.summary_id, EMBEDDING_SIM(s1.embedding, s2.embedding) AS sim
FROM summaries s1, summaries s2
WHERE s1.summary_id = 2 AND s2.summary_id != 2
ORDER BY sim DESC LIMIT 10;

--Most semantically similar to text entry. EMBED() called only once.
WITH q(vec) AS (SELECT EMBED('input text for testing'))
SELECT document_id, sim, text FROM (
  SELECT d.document_id, d.text,
    EMBEDDING_SIM(q.vec, d.embedding) AS sim
  FROM documents d, q
)
WHERE sim > 0.7
ORDER BY sim DESC LIMIT 10;
```

### `EMBED(text)`
Generates a sentence embedding via a GGUF model, loaded through **libllama**
(MacPorts `llama.cpp` package). Mean-pooled, L2-normalized, returned as a
BLOB — same shape as embeddings already stored in Ragger's DB.

SQLite extensions can't actually define new `PRAGMA`s (the syntax is
hardcoded into the core parser, not an extension point like functions/vtabs
are), so the closest equivalent — settings that persist in the database file
so you never re-type them — is a `semqlite_config` table plus two helper
functions:

```sql
SELECT SEMQLITE_SET('embedding_model', '/path/to/all-MiniLM-L12-v2'); -- or,
SELECT SEMQLITE_SET('embedding_model', (SELECT CONCAT('~/.ragger/models/', value) 
    FROM settings WHERE key = 'embedding_model'));

SELECT SEMQLITE_SET('embedding_dims', '384');            -- optional, 1..4096; omit/0 = model's native dim
SELECT SEMQLITE_SET('embedding_vector_type', 'f16');     -- "f16" (default), "f32", "bf16", or "int8"
SELECT SEMQLITE_SET('embedding_offset', '0');            -- bytes of zero-filled header to prepend (default 0)
SELECT SEMQLITE_SET('embedding_skip_renorm', '0');       -- set to 1 to skip L2 normalization (testing)

SELECT SEMQLITE_GET('embedding_model');   -- read back current setting

SELECT EMBED('some text to embed');     -- BLOB, per the settings above
```

Settings are set once per database and persist across `semqlite` restarts
(they live in `semqlite_config`, auto-created on first use). The model itself
is loaded lazily on first `EMBED()` call and cached for the process
lifetime, keyed by path — calling `SEMQLITE_SET('embedding_model', ...)` with
a different path swaps the cached model on the next call.

`embedding_vector_type` controls the on-disk blob format:
- **f32** — 4 bytes/dim, lossless
- **f16** — 2 bytes/dim, IEEE half precision (default, matches Ragger)
- **bf16** — 2 bytes/dim, bfloat16 (f32's exponent range, less mantissa)
- **int8** — 1 byte/dim + 2-byte f16 scale suffix (symmetric per-vector
  quantization, scale = max|x| / 127)

`embedding_offset` reserves N zero-filled bytes at the start of the blob,
allowing an external tool to write a header there. `EMBEDDING_SIM`/`DIST`
skip the same offset when decoding. Set both to the same value.

`embedding_skip_renorm` disables the L2 normalization step in `EMBED()` —
useful for testing whether normalization affects retrieval quality on your
data. Default is off (normalize).

`embedding_dims` smaller than the model's native output truncates
(Matryoshka-style slicing — not re-trained for it, just a slice, fine for
experimentation); larger zero-pads.

Any GGUF text-embedding model works with the `llama` backend (mean/CLS/last
pooling all handled via `llama_context_params.pooling_type =
LLAMA_POOLING_TYPE_MEAN`). The `onnx` backend (default) uses the same
tokenize → ONNX inference → mean-pool → L2-normalize pipeline as Ragger.

```sql
-- Sanity check: related text scores higher than unrelated text.
-- Absolute numbers depend on which model/backend produced the vectors —
-- not comparable across different models, only useful as a
-- related-vs-unrelated ordering check on a given model.
SELECT EMBEDDING_SIM(EMBED('cats are great pets'), EMBED('dogs are wonderful companions'));
SELECT EMBEDDING_SIM(EMBED('cats are great pets'), EMBED('quantum chromodynamics'));
```

Measured examples (yours will differ if you use a different model):

| Backend | Model | related sim | unrelated sim |
|---|---|---|---|
| `llama` | nomic-embed-text-v1.5 (768-dim) | ~0.49 | ~0.01 |
| `onnx` | all-MiniLM-L6-v2 (384-dim) | ~0.94 | ~0.88 |

## Opening Ragger's database

Copy-paste this block to configure `semqlite` for Ragger's `memories.db`.
It reads all embedding settings directly from Ragger's `settings` table — no
hardcoded values to keep in sync:

```sql
.open ~/.ragger/memories.db

-- Load embedding config from Ragger's settings table.
-- vector_type: f16 (default), f32, bf16, or int8
-- model:       resolved model directory name
-- dimensions:  384 for all-MiniLM-L12-v2, etc.
-- offset:      default 0; int8 scale is a 2-byte f16 suffix
--              after the payload, matching SemanticSQLite's convention.
SELECT SEMQLITE_SET('embedding_vector_type',
    (SELECT value FROM settings WHERE key = 'vector_type'));
SELECT SEMQLITE_SET('embedding_dims',
    (SELECT value FROM settings WHERE key = 'dimensions'));
SELECT SEMQLITE_SET('embedding_model',
    '~/.ragger/models/' || (SELECT value FROM settings WHERE key = 'embedding_model'));
SELECT SEMQLITE_SET('embedding_offset', '1');
```

After this, `EMBEDDING_SIM()`, `EMBEDDING_DIST()`, and `EMBED()` all work
against Ragger's live data with no further configuration.

## Build

### Requirements

| Requirement | Needed for | Notes |
|---|---|---|
| CMake ≥ 3.24 | everything | |
| C11 + C++20 compiler | everything | GCC 14 / Clang tested |
| `git` | everything | for the submodule below |
| **Rust toolchain (`cargo`)** | `-DSEMQLITE_ONNX=ON` (default) | `tokenizers-cpp`'s HuggingFace backend is a Rust crate built via `cargo build`, invoked automatically by CMake. **Not optional** unless you build with `-DSEMQLITE_ONNX=OFF`. See install command below. |
| `libllama` (llama.cpp) | `embedding_embedder='llama'` | MacPorts on macOS, build-from-source on Linux — see below. Not needed for the ONNX backend, but the CMake always looks for it (both backends can coexist in one binary). |
| Internet access (first build only) | `-DSEMQLITE_ONNX=ON` | ONNX Runtime is auto-fetched from GitHub releases; cached under `build/_deps/` after the first configure. |
| `libedit` (optional) | interactive shell usage | Gives the shell arrow-key/history line editing. Without it, arrow keys print raw escape sequences and there's no command history — the shell still works, it's just unpleasant to use interactively. macOS (MacPorts): `sudo port install libedit`. Debian/Linux: `sudo apt install libedit-dev`. Auto-detected; silently skipped if not found. |

If a requirement is missing, CMake usually fails at build time with an
opaque error rather than a clean message at configure time (e.g. missing
`cargo` shows up later as `no such file or directory` from a custom build
command) — check this table first if a build fails partway through.

```bash
git clone <this-repo-url> SemanticSQLite
cd SemanticSQLite
git submodule update --init --recursive   # pulls vendor/tokenizers-cpp + its deps
```

Install Rust (required for the default ONNX backend):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

If `cargo` gets installed *after* an earlier failed configure, run
`rm -rf build` before reconfiguring — CMake's `find_program()` caches a
"not found" result and won't recheck on a stale cache.

Install `libllama` (for the `llama` embedding backend — see
`embedding_embedder` above):

```bash
# Build from source
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=ON -DGGML_VULKAN=ON  # or -DGGML_CUDA=ON
cmake --build build -j$(nproc)
sudo cmake --install build
sudo ldconfig
```

The ONNX embedding backend (default) needs no separate install — ONNX
Runtime is auto-fetched at configure time for your platform (macOS
arm64/x86_64, Linux x86_64/aarch64), and `tokenizers-cpp` comes from the
submodule pulled above. Disable it with `--no-onnx` (build script) or
`-DSEMQLITE_ONNX=OFF` (manual cmake) if you only want the `llama` backend.

[c_lib](https://github.com/diskerror/c_lib) (shared C++ utilities) is
auto-fetched from GitHub at configure time via CMake FetchContent.

```bash
./scripts/build.sh              # check deps, configure, build
sudo cmake --install build      # installs semqlite to /usr/local/bin
```

### Dev build (local c_lib)

If you have a local checkout of [c_lib](https://github.com/diskerror/c_lib),
create a `CMakeUserPresets.json` (gitignored) to use it instead of fetching
from GitHub:

```json
{
  "version": 6,
  "configurePresets": [{
    "name": "dev",
    "inherits": "default",
    "cacheVariables": {
      "FETCHCONTENT_SOURCE_DIR_C_LIB": "/path/to/your/c_lib"
    }
  }]
}
```

The build script auto-detects this file and passes the variable to cmake.
CLion also picks up the `dev` preset from its CMake profile dropdown.

Manual cmake (without the build script):

```bash
cmake --preset dev              # or: cmake -B build
cmake --build build -j8
sudo cmake --install build
```

Produces `semqlite` — drop-in `sqlite3` replacement. Verified on macOS
(Apple Silicon) and Debian 13 (x86_64), both `-DSEMQLITE_ONNX=ON` and `OFF`.

## Usage

```bash
semqlite ~/.ragger/memories.db              # interactive shell
semqlite ~/.ragger/memories.db "SELECT ..."  # one-shot query
```

## Refreshing the vendored SQLite source

The amalgamation was generated from `sqlite/sqlite` at the commit recorded in
`vendor/sqlite/UPSTREAM_COMMIT.txt`:

```bash
git clone https://github.com/sqlite/sqlite
cd /path/to/sqlite-clone && git pull
mkdir build-amal && cd build-amal
../configure --disable-tcl
make sqlite3.c shell.c sqlite3.h
cp sqlite3.c sqlite3.h shell.c /path/to/SemanticSQLite/vendor/sqlite/
```

## License

SQLite itself is public domain. `DMPHON`/Double Metaphone is a published,
widely-implemented public algorithm. `STEM_PORTER` is the classic public-domain
1980 Porter algorithm. `STEM_SNOWBALL` links the vendored Snowball
`libstemmer_c` (BSD-3-Clause, [snowballstem.org](https://snowballstem.org)).
`EMBED()` links libllama (MIT license). This project's own code (the
`EMBEDDING_SIM`/`EMBEDDING_DIST`/`EMBED` implementations, the shell-integration
glue) is released with no restrictions — do whatever you want with it.
