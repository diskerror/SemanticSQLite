// double_metaphone_capi.h — plain-C bridge to the C++ Double Metaphone
// implementation (src/double_metaphone.cpp), so C translation units
// (ext_functions.c) can call it without pulling in C++ headers.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Phonize `text` per `mode`:
//   0 = all words, both codes (primary + alternate when they differ)
//   1 = first code per word (the primary — always present)
//   2 = last code per word (the alternate when one exists, else the
//       primary — always present, never NULL for alphabetic input)
// Returns a malloc'd, space-joined string (caller must semext_free() it),
// or NULL only for non-alphabetic/empty input.
char *semext_phonize_mode(const char *text, int mode);

// Free a string returned by semext_phonize_mode().
void semext_free(char *p);

#ifdef __cplusplus
}
#endif
