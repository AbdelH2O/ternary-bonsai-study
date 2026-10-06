// Compare an ordinary and a Hadamard-folded GGUF through the pinned llama runtime.
// Build with the matching llama.h and libllama.so, then run on the CPU.
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Run {
    std::vector<llama_token> tokens;
    std::vector<std::vector<float>> prompt_logits;
    std::vector<float> decode_logits;
    llama_token decode_token = -1;
};

static void must(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static Run eval(const char * model_path, const std::vector<llama_token> * fixed_tokens, llama_token fixed_decode) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    must(model != nullptr, "model load failed");
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = 128;
    cp.n_ubatch = 128;
    llama_context * ctx = llama_init_from_model(model, cp);
    must(ctx != nullptr, "context creation failed");
    llama_set_n_threads(ctx, 8, 8);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    Run result;
    const std::string prompt = "The quick brown fox jumps over the lazy dog.";
    result.tokens.resize(128);
    int32_t n = llama_tokenize(vocab, prompt.c_str(), int32_t(prompt.size()),
                               result.tokens.data(), int32_t(result.tokens.size()), true, false);
    must(n > 1 && n < 128, "tokenization failed");
    result.tokens.resize(n);
    if (fixed_tokens) must(result.tokens == *fixed_tokens, "tokenizer mismatch between models");

    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        batch.token[i] = result.tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    must(llama_decode(ctx, batch) == 0, "prefill decode failed");
    result.prompt_logits.reserve(n);
    for (int i = 0; i < n; ++i) {
        const float * row = llama_get_logits_ith(ctx, i);
        must(row != nullptr, "missing prefill logits");
        result.prompt_logits.emplace_back(row, row + n_vocab);
    }
    llama_batch_free(batch);

    const auto & last = result.prompt_logits.back();
    result.decode_token = fixed_decode >= 0 ? fixed_decode :
        llama_token(std::max_element(last.begin(), last.end()) - last.begin());
    batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 1;
    batch.token[0] = result.decode_token;
    batch.pos[0] = n;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;
    must(llama_decode(ctx, batch) == 0, "single-token decode failed");
    const float * row = llama_get_logits_ith(ctx, 0);
    must(row != nullptr, "missing decode logits");
    result.decode_logits.assign(row, row + n_vocab);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return result;
}

static void metrics(const std::vector<float> & a, const std::vector<float> & b) {
    must(a.size() == b.size(), "logit shape mismatch");
    double sum_abs = 0, sum_sq = 0, base_sq = 0, max_abs = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        must(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite logit");
        const double d = double(b[i]) - double(a[i]);
        sum_abs += std::abs(d);
        sum_sq += d*d;
        base_sq += double(a[i])*double(a[i]);
        max_abs = std::max(max_abs, std::abs(d));
    }
    const size_t top_a = std::max_element(a.begin(), a.end()) - a.begin();
    const size_t top_b = std::max_element(b.begin(), b.end()) - b.begin();
    std::cout << "{\"max_abs\":" << max_abs << ",\"mean_abs\":" << sum_abs/a.size()
              << ",\"relative_l2\":" << std::sqrt(sum_sq/base_sq)
              << ",\"top1_base\":" << top_a << ",\"top1_folded\":" << top_b << "}";
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr << "usage: compare_fp_logits base.gguf folded.gguf\n";
        return 2;
    }
    try {
        llama_backend_init();
        const Run base = eval(argv[1], nullptr, -1);
        const Run folded = eval(argv[2], &base.tokens, base.decode_token);
        must(base.prompt_logits.size() == folded.prompt_logits.size(), "prefill length mismatch");
        std::cout << std::setprecision(10) << "{\"token_ids\":[";
        for (size_t i = 0; i < base.tokens.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << base.tokens[i];
        }
        std::cout << "],\"forced_decode_token\":" << base.decode_token << ",\"prefill\":[";
        for (size_t i = 0; i < base.prompt_logits.size(); ++i) {
            if (i) std::cout << ',';
            metrics(base.prompt_logits[i], folded.prompt_logits[i]);
        }
        std::cout << "],\"decode\":";
        metrics(base.decode_logits, folded.decode_logits);
        std::cout << "}\n";
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "comparison failed: " << e.what() << '\n';
        return 1;
    }
}
