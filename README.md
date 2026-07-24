# SemanticSQLite

SQLite CLI (`sqlite-ext`) with custom functions for semantic search experimentation on Ragger databases.

## Custom Functions

### `DMPHON(text [, mode])`
Double Metaphone phonetic encoding.

- **mode 0** (default): returns all phonetic codes for all words, space-separated (both primary and alternate when they differ)
- **mode 1**: primary code(s) only
- **mode 2**: secondary code(s) only (NULL if no alternate exists)

```sql
SELECT DMPHON('Don''t panic');          -- 'TNT PNK'
SELECT DMPHON('Schmidt', 1);           -- 'XMT'
SELECT DMPHON('Schmidt', 2);           -- 'SMT'
```

### `EMBEDDING_SCORE(blob1, blob2)`
Cosine similarity between two embedding BLOBs. Auto-detects f16 vs f32 storage.

```sql
-- Find memories most similar to a reference embedding
SELECT text, EMBEDDING_SCORE(embedding, (SELECT embedding FROM summaries WHERE summary_id = 42))
FROM summaries
ORDER BY 2 DESC
LIMIT 10;
```

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

```bash
# Interactive REPL on a Ragger DB
sqlite-ext ~/.ragger/memory.db

# One-shot query
sqlite-ext ~/.ragger/memory.db -cmd "SELECT DMPHON('hello world')"
```
