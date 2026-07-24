// double_metaphone_capi.cpp — C-linkage bridge over double_metaphone.cpp's
// C++ API (namespace semext, std::string/std::vector), so plain-C callers
// (ext_functions.c) can use it without touching C++ headers directly.

#include "double_metaphone_capi.h"
#include "double_metaphone.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

// Mirrors semext::phonize()'s word-splitting, but picks a single code
// per word according to `mode` instead of always emitting both.
std::string phonize_mode(const char *text, int mode) {
    std::string out;
    std::string word;
    auto flush = [&]() {
        if (word.empty()) return;
        auto codes = semext::double_metaphone(word);
        if (codes.empty()) { word.clear(); return; }
        std::string picked;
        if (mode == 0) {
            for (size_t i = 0; i < codes.size(); ++i) {
                if (i > 0) picked += ' ';
                picked += codes[i];
            }
        } else if (mode == 1) {
            picked = codes.front();  // first code — always exists
        } else if (mode == 2) {
            picked = codes.back();  // last code — same as front() when only one exists
        }
        if (!picked.empty()) {
            if (!out.empty()) out += ' ';
            out += picked;
        }
        word.clear();
    };
    for (const unsigned char *p = reinterpret_cast<const unsigned char *>(text); *p; ++p) {
        if (std::isalpha(*p) || *p == '\'') word += static_cast<char>(*p);
        else flush();
    }
    flush();
    return out;
}

} // namespace

extern "C" char *semext_phonize_mode(const char *text, int mode) {
    if (!text) return nullptr;
    std::string result = phonize_mode(text, mode);
    if (result.empty()) return nullptr;
    char *out = static_cast<char *>(std::malloc(result.size() + 1));
    if (!out) return nullptr;
    std::memcpy(out, result.c_str(), result.size() + 1);
    return out;
}

extern "C" void semext_free(char *p) {
    std::free(p);
}
