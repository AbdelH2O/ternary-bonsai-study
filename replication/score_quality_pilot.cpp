// Position-matched teacher-forced NLL for the frozen small hybrid screen.
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static void must(bool ok, const char * msg) {
    if (!ok) throw std::runtime_error(msg);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::cerr << "usage: score_quality_pilot model.gguf slice.txt [slice.txt ...]\n";
        return 2;
    }
    try {
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        llama_model * model = llama_model_load_from_file(argv[1], mp);
        must(model != nullptr, "model load failed");
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 128;
        cp.n_batch = 128;
        cp.n_ubatch = 128;
        llama_context * ctx = llama_init_from_model(model, cp);
        must(ctx != nullptr, "context creation failed");
        llama_set_n_threads(ctx, 8, 8);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int vocab_size = llama_vocab_n_tokens(vocab);
        std::cout << std::setprecision(12);
        for (int file_index = 2; file_index < argc; ++file_index) {
            std::ifstream stream(argv[file_index], std::ios::binary);
            must(bool(stream), "slice open failed");
            const std::string content((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
            std::vector<llama_token> tokens(2048);
            int n = llama_tokenize(vocab, content.c_str(), int(content.size()),
                                   tokens.data(), int(tokens.size()), true, false);
            must(n >= 128 && n <= int(tokens.size()), "slice has fewer than 128 tokens or buffer exceeded");
            tokens.resize(128);
            llama_memory_clear(llama_get_memory(ctx), true);
            llama_batch batch = llama_batch_init(127, 0, 1);
            batch.n_tokens = 127;
            for (int i = 0; i < 127; ++i) {
                batch.token[i] = tokens[i];
                batch.pos[i] = i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = i >= 63;
            }
            must(llama_decode(ctx, batch) == 0, "decode failed");
            double sum = 0;
            for (int target = 64; target < 128; ++target) {
                const float * row = llama_get_logits_ith(ctx, target - 1);
                must(row != nullptr, "missing logit row");
                const double max_logit = *std::max_element(row, row + vocab_size);
                double denominator = 0;
                for (int j = 0; j < vocab_size; ++j) denominator += std::exp(double(row[j]) - max_logit);
                const double nll = max_logit + std::log(denominator) - row[tokens[target]];
                must(std::isfinite(nll), "nonfinite NLL");
                sum += nll;
            }
            llama_batch_free(batch);
            std::cout << "{\"file\":\"" << argv[file_index] << "\",\"nll\":" << sum / 64
                      << ",\"tokens\":[";
            for (int i = 0; i < 128; ++i) {
                if (i) std::cout << ',';
                std::cout << tokens[i];
            }
            std::cout << "]}" << std::endl;
        }
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "scoring failed: " << error.what() << '\n';
        return 1;
    }
}
