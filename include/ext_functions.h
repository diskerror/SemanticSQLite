// ext_functions.h — custom SQL function registration for sqlite-ext.
#pragma once

#include <sqlite3.h>

namespace semext {

// Register all custom SQL functions on the given db connection.
//   DMPHON(text)            — returns primary DMetaPhone code
//   DMPHON(text, mode)      — mode: 0=both (default), 1=primary only, 2=secondary only
//   EMBEDDING_SCORE(b1, b2) — cosine similarity between two embedding blobs (f16 or f32)
void register_functions(sqlite3* db);

} // namespace semext
