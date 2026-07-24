// double_metaphone.h — phonetic encoding (standalone, no Ragger deps).
// Forked from Ragger's ragger/double_metaphone.h.
#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace semext {

// Compute Double Metaphone codes for a single word.
// Returns {primary} or {primary, alternate}.
std::vector<std::string> double_metaphone(std::string_view word,
                                          size_t max_length = 4);

// Phonize text into space-joined DMetaPhone codes.
std::string phonize(std::string_view text);

} // namespace semext
