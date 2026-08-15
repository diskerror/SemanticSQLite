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
The storage dtype and byte layout are controlled by `semext_config` settings:

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
their behavior depends on the mutable `semext_config` settings. Same blob
inputs with a different `embedding_vector_type` produce different results.

```sql
-- Configure for Ragger's default: raw f16 blobs, no header
SELECT SEMEXT_SET('embedding_vector_type', 'f16');
SELECT SEMEXT_SET('embedding_offset', '0');

-- Most semantically similar summaries to summary_id=2
SELECT s2.summary_id, EMBEDDING_SIM(s1.embedding, s2.embedding) AS sim
FROM summaries s1, summaries s2
WHERE s1.summary_id = 2 AND s2.summary_id != 2
ORDER BY sim DESC LIMIT 10;

-- If your blobs have a 12-byte header (e.g. Ragger's vector_codec format)
SELECT SEMEXT_SET('embedding_offset', '12');
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
SELECT SEMEXT_SET('embedding_vector_type', 'f16');     -- "f16" (default), "f32", "bf16", or "int8"
SELECT SEMEXT_SET('embedding_offset', '0');            -- bytes of zero-filled header to prepend (default 0)
SELECT SEMEXT_SET('embedding_skip_renorm', '0');       -- set to 1 to skip L2 normalization (testing)

SELECT SEMEXT_GET('embedding_model');   -- read back current setting

SELECT EMBED('some text to embed');     -- BLOB, per the settings above
```

Settings are set once per database and persist across `sqlite-ext` restarts
(they live in `semext_config`, auto-created on first use). The model itself
is loaded lazily on first `EMBED()` call and cached for the process
lifetime, keyed by path — calling `SEMEXT_SET('embedding_model', ...)` with
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

Any GGUF text-embedding model works (mean/CLS/last pooling all handled via
`llama_context_params.pooling_type = LLAMA_POOLING_TYPE_MEAN`). Verified
against `nomic-embed-text-v1.5` (768-dim, nomic-bert architecture).

```sql
-- Sanity check: related text scores higher than unrelated text
SELECT EMBEDDING_SIM(EMBED('cats are great pets'), EMBED('dogs are wonderful companions')); -- ~0.49
SELECT EMBEDDING_SIM(EMBED('cats are great pets'), EMBED('quantum chromodynamics'));        -- ~0.01
```

## Build

```bash
git clone <this-repo-url> SemanticSQLite
cd SemanticSQLite
git submodule update --init --recursive   # pulls vendor/tokenizers-cpp + its deps
```

Requires `libllama` (for the `llama` embedding backend — see
`embedding_embedder` above):

```bash
# macOS (MacPorts)
sudo port install llama.cpp

# Debian/Linux — build from source
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=ON -DGGML_VULKAN=ON  # or -DGGML_CUDA=ON
cmake --build build -j$(nproc)
sudo cmake --install build
sudo ldconfig
```

The ONNX embedding backend (default) needs no separate install — ONNX
Runtime is auto-fetched at configure time for your platform (macOS
arm64/x86_64, Linux x86_64/aarch64), and `tokenizers-cpp` comes from the
submodule pulled above. Disable it with `-DSEMEXT_ONNX=OFF` if you only
want the `llama` backend (skips the ONNX Runtime download entirely).

```bash
mkdir build && cd build
cmake ..                    # add -DSEMEXT_ONNX=OFF to skip ONNX Runtime
make -j$(nproc)
sudo cmake --install .       # installs sqlite-ext to /usr/local/bin
```

Produces `sqlite-ext` — drop-in `sqlite3` replacement. Developed and
verified on macOS (Apple Silicon); the CMake build targets Debian/Linux
too (RPATH handling for non-standard lib install locations, platform
detection for the ONNX Runtime download) but hasn't been build-tested
there yet.

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
git clone https://github.com/sqlite/sqlite
cd /path/to/sqlite-clone && git pull
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
