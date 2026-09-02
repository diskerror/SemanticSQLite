// embed.h — EMBED(text) SQL function: generates a sentence embedding via a
// GGUF model loaded through libllama (mean-pooled, L2-normalized). Model
// path / target dimension / storage dtype are read from the semqlite_config
// table (see semqlite_config_get/set below) rather than hardcoded — SQLite
// doesn't support extension-defined PRAGMAs, so a config table is the
// closest equivalent that "sticks" per-database without re-typing.
#pragma once

#include <sqlite3.h>

// Registers EMBED(text), SEMQLITE_SET(key, value), SEMQLITE_GET(key) on db.
// Also ensures the semqlite_config table exists.
int semqlite_register_embed(sqlite3 *db);

// Frees any cached libllama model/context. Call once at process exit
// (the shell doesn't have a clean unload hook, so this is best-effort;
// omitting it just leaks until process exit, which is harmless for a CLI).
void semqlite_embed_shutdown(void);
