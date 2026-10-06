// Build against Prism llama.cpp 842b188. Split prefill at the answer boundary to match state_gate.
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::string slurp(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

static std::vector<llama_token> read_ids(const std::string & path) {
    std::istringstream in(slurp(path));
    std::vector<llama_token> ids;
    llama_token id;
    while (in >> id) ids.push_back(id);
    if (!in.eof()) throw std::runtime_error("invalid token-ID file: " + path);
    return ids;
}

static double gold_nll(const float * logits, int vocab_size, llama_token gold) {
    if (!logits || gold < 0 || gold >= vocab_size) throw std::runtime_error("invalid gold logit");
    float max_logit = logits[0];
    for (int j = 1; j < vocab_size; ++j) max_logit = std::max(max_logit, logits[j]);
    double sum = 0;
    for (int j = 0; j < vocab_size; ++j) sum += std::exp(double(logits[j] - max_logit));
    return double(max_logit) + std::log(sum) - double(logits[gold]);
}

int main(int argc, char ** argv) {
    try {
        if (argc < 2) throw std::runtime_error("usage: split_scorer tokenize MODEL TEXT IDS | score MODEL CASES_TSV OUTPUT_JSONL BATCH UBATCH");
        const std::string mode = argv[1];
        if ((mode == "tokenize" && argc != 5) || (mode == "score" && argc != 7)) throw std::runtime_error("wrong argument count");
        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.vocab_only = mode == "tokenize";
        model_params.n_gpu_layers = mode == "score" ? 99 : 0;
        llama_model * model = llama_model_load_from_file(argv[2], model_params);
        if (!model) throw std::runtime_error("model load failed");
        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (mode == "tokenize") {
            const std::string data = slurp(argv[3]);
            int32_t capacity = std::max<int32_t>(4096, data.size() / 2);
            std::vector<llama_token> ids(capacity);
            int32_t n = llama_tokenize(vocab, data.data(), data.size(), ids.data(), capacity, false, false);
            if (n < 0) {
                ids.resize(-n);
                n = llama_tokenize(vocab, data.data(), data.size(), ids.data(), ids.size(), false, false);
            }
            if (n < 0) throw std::runtime_error("tokenization failed");
            std::ofstream out(argv[4]);
            for (int32_t i = 0; i < n; ++i) out << ids[i] << '\n';
            std::cerr << "tokenized " << n << " IDs; add_special=false; parse_special=false\n";
        } else if (mode == "score") {
            const int n_batch = std::stoi(argv[5]);
            const int n_ubatch = std::stoi(argv[6]);
            if (n_batch <= 0 || n_ubatch <= 0 || n_ubatch > n_batch) throw std::runtime_error("bad batch settings");
            auto params = llama_context_default_params();
            params.n_ctx = 16384;
            params.n_seq_max = 1;
            params.n_batch = n_batch;
            params.n_ubatch = n_ubatch;
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            llama_context * ctx = llama_init_from_model(model, params);
            if (!ctx) throw std::runtime_error("context creation failed");
            llama_batch batch = llama_batch_init(n_batch, 0, 1);
            std::ifstream cases(argv[3]);
            std::ofstream out(argv[4]);
            if (!cases || !out) throw std::runtime_error("cannot open cases or output");
            const int vocab_size = llama_vocab_n_tokens(vocab);
            std::string line;
            while (std::getline(cases, line)) {
                if (line.empty()) continue;
                std::istringstream row(line);
                std::string case_id, ids_path;
                int p, h, m;
                if (!(row >> case_id >> ids_path >> p >> h >> m)) throw std::runtime_error("malformed case: " + line);
                const auto ids = read_ids(ids_path);
                if (h < 1 || m < 1 || p < h || size_t(p + m) > ids.size() || h + m > 16384) throw std::runtime_error("invalid span: " + case_id);
                llama_memory_clear(llama_get_memory(ctx), true);
                const auto t0 = std::chrono::steady_clock::now();
                std::vector<double> nll;
                nll.reserve(m);
                for (int offset = 0; offset < h; offset += n_batch) {
                    const int count = std::min(n_batch, h - offset);
                    batch.n_tokens = count;
                    for (int k = 0; k < count; ++k) {
                        batch.token[k] = ids[p - h + offset + k];
                        batch.pos[k] = offset + k;
                        batch.n_seq_id[k] = 1;
                        batch.seq_id[k][0] = 0;
                        batch.logits[k] = offset + k == h - 1;
                    }
                    if (llama_decode(ctx, batch) != 0) throw std::runtime_error("llama_decode failed: " + case_id);
                    if (offset + count == h) {
                        nll.push_back(gold_nll(llama_get_logits_ith(ctx, count - 1), vocab_size, ids[p]));
                    }
                }
                for (int offset = 0; offset < m - 1; offset += n_batch) {
                    const int count = std::min(n_batch, m - 1 - offset);
                    batch.n_tokens = count;
                    for (int k = 0; k < count; ++k) {
                        batch.token[k] = ids[p + offset + k];
                        batch.pos[k] = h + offset + k;
                        batch.n_seq_id[k] = 1;
                        batch.seq_id[k][0] = 0;
                        batch.logits[k] = true;
                    }
                    if (llama_decode(ctx, batch) != 0) throw std::runtime_error("continuation failed: " + case_id);
                    for (int k = 0; k < count; ++k) {
                        nll.push_back(gold_nll(llama_get_logits_ith(ctx, k), vocab_size, ids[p + offset + k + 1]));
                    }
                }
                llama_synchronize(ctx);
                if (int(nll.size()) != m) throw std::runtime_error("scored token count mismatch");
                const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                out << std::setprecision(12) << "{\"case_id\":\"" << case_id << "\",\"p\":" << p
                    << ",\"h\":" << h << ",\"m\":" << m << ",\"n_batch\":" << n_batch
                    << ",\"n_ubatch\":" << n_ubatch << ",\"seconds\":" << seconds << ",\"nll\":[";
                for (int i = 0; i < m; ++i) out << (i ? "," : "") << nll[i];
                out << "]}\n";
                out.flush();
                std::cerr << case_id << " h=" << h << " m=" << m << " seconds=" << seconds << '\n';
            }
            llama_batch_free(batch);
            llama_free(ctx);
        } else throw std::runtime_error("unknown mode");
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
