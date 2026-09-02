// embed_onnx.cpp — ONNX Runtime + tokenizers-cpp embedding backend.
// Mirrors Ragger's embedder.cpp pipeline (mean pooling, L2-normalize)
// so vectors are byte-compatible with Ragger's ONNX-generated embeddings.

#include "embed_onnx.h"

#include <onnxruntime_cxx_api.h>
#include <tokenizers_cpp.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// -----------------------------------------------------------------------
// Cached model state (one model at a time, like the llama backend).
// -----------------------------------------------------------------------
static struct {
    std::string                             path;
    std::unique_ptr<Ort::Env>               env;
    std::unique_ptr<Ort::Session>           session;
    std::unique_ptr<tokenizers::Tokenizer>  tokenizer;
    int                                     n_embd = 0;
} g_onnx;

static const char *g_last_error = nullptr;

// -----------------------------------------------------------------------
// UTF-8 sanitizer — replace invalid bytes with U+FFFD so the Rust
// tokenizer doesn't panic (same logic as Ragger's tokenizer_wrapper.cpp).
// -----------------------------------------------------------------------
static std::string sanitize_utf8(const std::string &s) {
    static constexpr char kRepl[] = "\xEF\xBF\xBD";
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) { out += static_cast<char>(c); ++i; continue; }
        int len = 0; unsigned char mask = 0; unsigned int cp_min = 0;
        if      ((c & 0xE0) == 0xC0) { len = 2; mask = 0x1F; cp_min = 0x80; }
        else if ((c & 0xF0) == 0xE0) { len = 3; mask = 0x0F; cp_min = 0x800; }
        else if ((c & 0xF8) == 0xF0) { len = 4; mask = 0x07; cp_min = 0x10000; }
        else { out += kRepl; ++i; continue; }
        if (i + (size_t)len > n) { out += kRepl; ++i; continue; }
        unsigned int cp = c & mask;
        bool ok = true;
        for (int k = 1; k < len; ++k) {
            unsigned char cc = static_cast<unsigned char>(s[i + k]);
            if ((cc & 0xC0) != 0x80) { ok = false; break; }
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (ok && (cp < cp_min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)))
            ok = false;
        if (ok) { out.append(s, i, (size_t)len); i += (size_t)len; }
        else    { out += kRepl; ++i; }
    }
    return out;
}

// -----------------------------------------------------------------------
extern "C" int semqlite_onnx_load(const char *model_dir, const char **errmsg) {
    *errmsg = nullptr;
    if (!model_dir || model_dir[0] == '\0') {
        *errmsg = "embedding_model not set — call SEMQLITE_SET('embedding_model', '/path/to/model_dir') first";
        return -1;
    }
    // Already loaded?
    if (!g_onnx.path.empty() && g_onnx.path == model_dir && g_onnx.session)
        return 0;

    // Unload previous
    g_onnx.session.reset();
    g_onnx.tokenizer.reset();
    g_onnx.env.reset();
    g_onnx.path.clear();
    g_onnx.n_embd = 0;

    fs::path dir(model_dir);
    // HF-exported ONNX models are laid out inconsistently: some repos put
    // model.onnx directly in the model dir, others nest it under onnx/
    // (the ONNX export subfolder convention used by optimum/transformers.js).
    // Try both.
    fs::path model_path = dir / "model.onnx";
    if (!fs::exists(model_path)) {
        fs::path nested = dir / "onnx" / "model.onnx";
        if (fs::exists(nested)) model_path = nested;
    }
    fs::path tok_path   = dir / "tokenizer.json";

    if (!fs::exists(model_path)) {
        g_last_error = "model.onnx not found (looked in model dir and model dir/onnx/)";
        *errmsg = g_last_error;
        return -1;
    }
    if (!fs::exists(tok_path)) {
        g_last_error = "tokenizer.json not found in the specified model directory";
        *errmsg = g_last_error;
        return -1;
    }

    try {
        // Load tokenizer
        std::ifstream f(tok_path);
        std::stringstream buf;
        buf << f.rdbuf();
        std::string json = buf.str();
        if (json.empty()) {
            *errmsg = "tokenizer.json is empty";
            return -1;
        }
        auto tok = tokenizers::Tokenizer::FromBlobJSON(json);
        if (!tok) {
            *errmsg = "failed to parse tokenizer.json";
            return -1;
        }

        // Load ONNX model
        auto env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "semqlite");
        Ort::SessionOptions opts;
        auto session = std::make_unique<Ort::Session>(*env, model_path.c_str(), opts);

        // Probe output shape to get embedding dimension: run a dummy
        // single-token input and read output_shape[2].
        // (Alternatively we could read model metadata, but this is reliable.)
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> dummy_ids = {0};  // CLS or pad token
        std::vector<int64_t> dummy_mask = {1};
        std::vector<int64_t> dummy_type = {0};
        std::vector<int64_t> shape = {1, 1};
        const char *in_names[]  = {"input_ids", "attention_mask", "token_type_ids"};
        const char *out_names[] = {"last_hidden_state"};
        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, dummy_ids.data(), 1, shape.data(), 2));
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, dummy_mask.data(), 1, shape.data(), 2));
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, dummy_type.data(), 1, shape.data(), 2));
        auto outputs = session->Run(Ort::RunOptions{nullptr}, in_names, inputs.data(), 3, out_names, 1);
        auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int n_embd = (out_shape.size() == 3) ? (int)out_shape[2] : 0;
        if (n_embd <= 0) {
            *errmsg = "failed to determine embedding dimensions from ONNX model";
            return -1;
        }

        g_onnx.env = std::move(env);
        g_onnx.session = std::move(session);
        g_onnx.tokenizer = std::move(tok);
        g_onnx.path = model_dir;
        g_onnx.n_embd = n_embd;
        return 0;

    } catch (const std::exception &e) {
        // Store the message so it survives the catch scope.
        static thread_local std::string err_buf;
        err_buf = e.what();
        *errmsg = err_buf.c_str();
        return -1;
    }
}

extern "C" int semqlite_onnx_encode(const char *text, int text_len,
                                   float **out, int *out_dims) {
    *out = nullptr;
    *out_dims = 0;

    if (!g_onnx.session || !g_onnx.tokenizer) return -1;

    try {
        std::string clean = sanitize_utf8(std::string(text, (size_t)text_len));
        std::vector<int32_t> ids = g_onnx.tokenizer->Encode(clean);
        if (ids.empty()) return -1;

        size_t seq_len = ids.size();
        std::vector<int64_t> input_ids(ids.begin(), ids.end());
        std::vector<int64_t> attention_mask(seq_len, 1);
        std::vector<int64_t> token_type_ids(seq_len, 0);
        std::vector<int64_t> shape = {1, (int64_t)seq_len};

        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const char *in_names[]  = {"input_ids", "attention_mask", "token_type_ids"};
        const char *out_names[] = {"last_hidden_state"};
        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, input_ids.data(), seq_len, shape.data(), 2));
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, attention_mask.data(), seq_len, shape.data(), 2));
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, token_type_ids.data(), seq_len, shape.data(), 2));

        auto outputs = g_onnx.session->Run(Ort::RunOptions{nullptr},
                                            in_names, inputs.data(), 3,
                                            out_names, 1);

        float *data = outputs[0].GetTensorMutableData<float>();
        auto os = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        if (os.size() != 3 || os[0] != 1) return -1;
        size_t out_seq = (size_t)os[1];
        size_t hidden  = (size_t)os[2];

        // Mean pooling with attention mask (all 1s here, but correct pattern)
        float *pooled = (float *)calloc(hidden, sizeof(float));
        if (!pooled) return -1;
        float mask_sum = 0.0f;
        for (size_t i = 0; i < out_seq; i++) {
            float m = (i < seq_len) ? (float)attention_mask[i] : 0.0f;
            mask_sum += m;
            for (size_t j = 0; j < hidden; j++)
                pooled[j] += data[i * hidden + j] * m;
        }
        if (mask_sum > 0.0f) {
            for (size_t j = 0; j < hidden; j++) pooled[j] /= mask_sum;
        }

        // L2-normalize
        float norm = 0.0f;
        for (size_t j = 0; j < hidden; j++) norm += pooled[j] * pooled[j];
        norm = std::sqrt(norm);
        if (norm > 1e-12f) {
            float inv = 1.0f / norm;
            for (size_t j = 0; j < hidden; j++) pooled[j] *= inv;
        }

        *out = pooled;
        *out_dims = (int)hidden;
        return 0;

    } catch (...) {
        return -1;
    }
}

extern "C" void semqlite_onnx_shutdown(void) {
    g_onnx.session.reset();
    g_onnx.tokenizer.reset();
    g_onnx.env.reset();
    g_onnx.path.clear();
    g_onnx.n_embd = 0;
}
