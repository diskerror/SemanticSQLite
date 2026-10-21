/**
 * @file embed.h
 * @brief EMBED(text) SQL function: generates a sentence embedding via a
 * GGUF model loaded through libllama (mean-pooled, L2-normalized).
 *
 * Model path / target dimension / storage dtype are read from the
 * semqlite_config table (see semqlite_config_get/set below) rather than
 * hardcoded — SQLite doesn't support extension-defined PRAGMAs, so a config
 * table is the closest equivalent that "sticks" per-database without
 * re-typing.
 */
#pragma once

#include <sqlite3.h>

/**
 * @brief Registers EMBED(text), SEMQLITE_SET(key, value), SEMQLITE_GET(key)
 * and SEMQLITE_SAVE() on db.
 *
 * Settings live in temp.semqlite_config (RAM); if the file has a saved
 * main.semqlite_config it is loaded at connection setup. Opening a database
 * never writes to it.
 *
 * @param db Open connection.
 * @return SQLITE_OK on success.
 */
int semqlite_register_embed(sqlite3 *db);

/**
 * @brief Reads a setting (malloc'd, caller frees).
 *
 * A known key that is missing is created with its default and returned; an
 * unknown missing key returns NULL.
 *
 * @param db  Open connection.
 * @param key Config key, e.g. "embedding_model".
 * @return malloc'd value, or NULL for an unknown missing key.
 */
char *semqlite_config_get(sqlite3 *db, const char *key);

/**
 * @brief Frees any cached libllama model/context.
 *
 * Call once at process exit (the shell doesn't have a clean unload hook, so
 * this is best-effort; omitting it just leaks until process exit, which is
 * harmless for a CLI).
 */
void semqlite_embed_shutdown(void);

/**
 * @brief Set the directory under which relative embedding_model ids are
 * resolved (CLI --model-directory).
 *
 * @param dir Directory; NULL or "" restores the default (\<DB dir\>/models).
 */
void semqlite_set_model_directory(const char *dir);
