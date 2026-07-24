// main.cpp — sqlite-ext: SQLite CLI shell with custom functions.
//
// Wraps SQLite's built-in shell (sqlite3_shell.c) with DMPHON() and
// EMBEDDING_SCORE() pre-registered. If sqlite3_shell.c isn't available,
// falls back to a minimal REPL.

#include "ext_functions.h"

#include <sqlite3.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

// Minimal interactive REPL when we can't link the full shell.
static int mini_repl(sqlite3* db) {
    std::string line;
    std::string sql;
    const char* prompt = "sqlite-ext> ";
    const char* cont   = "       ...> ";

    std::cout << "sqlite-ext (SQLite " << sqlite3_libversion()
              << ") — custom functions: DMPHON(), EMBEDDING_SCORE()\n"
              << "Enter \".quit\" to exit.\n";

    while (true) {
        std::cout << (sql.empty() ? prompt : cont) << std::flush;
        if (!std::getline(std::cin, line)) break;

        // Dot-commands
        if (sql.empty() && !line.empty() && line[0] == '.') {
            if (line == ".quit" || line == ".exit") break;
            if (line == ".help") {
                std::cout << ".quit       Exit\n"
                          << ".tables     List tables\n"
                          << ".schema     Show schema\n";
                continue;
            }
            if (line == ".tables") {
                line = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1;";
            } else if (line == ".schema") {
                line = "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY 1;";
            } else {
                std::cout << "Unknown command: " << line << "\n";
                continue;
            }
        }

        sql += line;
        sql += '\n';

        if (!sqlite3_complete(sql.c_str())) continue;

        char* errmsg = nullptr;
        int rc = sqlite3_exec(db, sql.c_str(),
            [](void*, int ncols, char** vals, char** cols) -> int {
                for (int i = 0; i < ncols; i++) {
                    if (i > 0) std::cout << '|';
                    std::cout << (vals[i] ? vals[i] : "NULL");
                }
                std::cout << '\n';
                return 0;
            }, nullptr, &errmsg);

        if (rc != SQLITE_OK) {
            std::cerr << "Error: " << (errmsg ? errmsg : "unknown") << '\n';
            sqlite3_free(errmsg);
        }
        sql.clear();
    }
    return 0;
}

int main(int argc, char** argv) {
    // Determine DB path from argv (first non-flag arg, or :memory:)
    const char* db_path = nullptr;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') { db_path = argv[i]; break; }
    }

    sqlite3* db = nullptr;
    int rc = sqlite3_open(db_path ? db_path : ":memory:", &db);
    if (rc != SQLITE_OK) {
        std::cerr << "Cannot open database: " << sqlite3_errmsg(db) << '\n';
        sqlite3_close(db);
        return 1;
    }

    // Register our custom functions
    semext::register_functions(db);

    // If we have remaining SQL args, execute them and exit
    // Otherwise, run the REPL
    bool has_sql_arg = false;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "-cmd") == 0 && i + 1 < argc) {
            char* errmsg = nullptr;
            rc = sqlite3_exec(db, argv[i + 1], [](void*, int ncols, char** vals, char**) -> int {
                for (int j = 0; j < ncols; j++) {
                    if (j > 0) std::cout << '|';
                    std::cout << (vals[j] ? vals[j] : "NULL");
                }
                std::cout << '\n';
                return 0;
            }, nullptr, &errmsg);
            if (rc != SQLITE_OK) {
                std::cerr << "Error: " << (errmsg ? errmsg : "unknown") << '\n';
                sqlite3_free(errmsg);
            }
            has_sql_arg = true;
            i++; // skip the SQL string
        }
    }

    if (!has_sql_arg) {
        rc = mini_repl(db);
    }

    sqlite3_close(db);
    return rc;
}
