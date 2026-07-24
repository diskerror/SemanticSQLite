// double_metaphone_capi.h — plain-C bridge to the C++ Double Metaphone
// implementation (src/double_metaphone.cpp), so C translation units
// (ext_functions.c) can call it without pulling in C++ headers.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Phonize `text` per `mode`:
//   0 = all words, both codes (primary + alternate when they differ)
//   1 = primary code(s) only
//   2 = secondary code(s) only
// Returns a malloc'd, space-joined string (caller must semext_free() it),
// or NULL if the result would be empty (e.g. mode 2 with no alternates,
// or non-alphabetic input).
char *semext_phonize_mode(const char *text, int mode);

// Free a string returned by semext_phonize_mode().
void semext_free(char *p);

#ifdef __cplusplus
}
#endif
