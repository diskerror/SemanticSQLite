#!/bin/sh
# Usage: test_llama_loader.sh /path/to/semqlite
# libllama is loaded lazily: a missing or wrong library must become an SQL
# error from EMBED() only, never stop the shell or other functions.
S="$1"
[ -x "$S" ] || { echo "usage: $0 /path/to/semqlite"; exit 2; }
T=$(mktemp -d) || exit 2
trap 'rm -rf "$T"' EXIT
: > "$T/x.gguf"            # exists, so the "model not found" check passes
fail=0
check() { if [ "$2" = ok ]; then echo "ok   - $1"; else echo "FAIL - $1"; fail=1; fi; }
embed_sql="select semqlite_set('embedding_embedder','llama'); select semqlite_set('embedding_model','$T/x.gguf'); select embed('hi');"

# 1. Missing library: EMBED() errors, message names the file, says what to do.
out=$(SEMQLITE_LLAMA_LIB="$T/libllama.0.dylib" "$S" :memory: "$embed_sql" 2>&1)
case $out in *"cannot load $T/libllama.0.dylib"*"Rebuild SemanticSQLite"*) check "missing lib -> clear EMBED error" ok;; *) check "missing lib -> clear EMBED error: $out" bad;; esac

# 2. Wrong library (a file with no llama symbols): names the missing symbol.
: > "$T/fake.dylib"
out=$(SEMQLITE_LLAMA_LIB="$T/fake.dylib" "$S" :memory: "$embed_sql" 2>&1)
case $out in *"cannot load"*|*"has no symbol"*) check "wrong lib -> EMBED error" ok;; *) check "wrong lib -> EMBED error: $out" bad;; esac

# 3. Everything else still works with the library missing (exit 0, right value).
out=$(SEMQLITE_LLAMA_LIB="$T/nope" "$S" :memory: "select 1+1, stem_porter('running');" 2>&1); rc=$?
[ $rc -eq 0 ] && [ "$out" = "2|run" ] && check "non-EMBED SQL works without libllama" ok || check "non-EMBED SQL works without libllama: rc=$rc $out" bad

# 4. -version reports the library status line.
out=$("$S" -version 2>&1)
case $out in *"libllama:"*) check "-version shows libllama status" ok;; *) check "-version shows libllama status" bad;; esac

# 5. The binary has no load-time dependency on libllama or libggml.
if command -v otool >/dev/null 2>&1; then deps=$(otool -L "$S"); else deps=$(ldd "$S" 2>/dev/null); fi
case $deps in *libllama*|*libggml*) check "no link-time libllama/libggml dependency" bad;; *) check "no link-time libllama/libggml dependency" ok;; esac
exit $fail
