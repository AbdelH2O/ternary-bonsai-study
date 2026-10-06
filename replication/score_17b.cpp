// Token-ID scorer for the frozen 1.7B comparison (results/q17b/protocol.json).
//
//   score_17b MODEL CASES.tsv OUT.jsonl GPU_LAYERS NORM_VOCAB ROPE
//
// NORM_VOCAB: softmax over token IDs [0, NORM_VOCAB) only, so arms with padded embedding rows
// (Qwen3: 151936) and trimmed ones (Prism: 151669) share one normalizer. ROPE: "model" uses the
// file's rope scaling; "none" forces LLAMA_ROPE_SCALING_TYPE_NONE (disables YaRN).
//
// Each case line is tab-separated: id, kind, token IDs (space-separated), argument.
//   kind "nll":    argument = first target position t; prints the teacher-forced NLL of every
//                  token at positions t..n-1 (nat), in order.
//   kind "choice": argument = comma-separated candidate token IDs; prints the log-probability of
//                  each candidate as the next token after the whole sequence (full-vocabulary softmax).
// Memory is cleared between cases. No tokenization happens here: every arm sees identical IDs.
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void must(bool ok, const std::string & msg) {
    if (!ok) throw std::runtime_error(msg);
}

static std::vector<int> ints(const std::string & text, char sep) {
    std::vector<int> out;
    std::stringstream in(text);
    std::string item;
    while (std::getline(in, item, sep)) {
        if (!item.empty()) out.push_back(std::stoi(item));
    }
    return out;
}

static double log_normalizer(const float * row, int vocab_size) {
    const double max_logit = *std::max_element(row, row + vocab_size);
    double sum = 0;
    for (int j = 0; j < vocab_size; ++j) sum += std::exp(double(row[j]) - max_logit);
    return max_logit + std::log(sum);
}

int main(int argc, char ** argv) {
    if (argc != 7 || (std::string(argv[6]) != "model" && std::string(argv[6]) != "none")) {
        std::cerr << "usage: score_17b MODEL CASES.tsv OUT.jsonl GPU_LAYERS NORM_VOCAB model|none\n";
        return 2;
    }
    try {
        const int n_ctx = 2048;
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = std::stoi(argv[4]);
        llama_model * model = llama_model_load_from_file(argv[1], mp);
        must(model != nullptr, "model load failed");
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n_ctx;
        cp.n_batch = n_ctx;
        cp.n_ubatch = 512;
        cp.n_seq_max = 1;
        if (std::string(argv[6]) == "none") cp.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
        llama_context * ctx = llama_init_from_model(model, cp);
        must(ctx != nullptr, "context creation failed");
        llama_set_n_threads(ctx, 8, 8);
        const int vocab_size = std::stoi(argv[5]);
        must(vocab_size > 0 && vocab_size <= llama_vocab_n_tokens(llama_model_get_vocab(model)), "bad NORM_VOCAB");
        llama_batch batch = llama_batch_init(n_ctx, 0, 1);
        std::ifstream cases(argv[2]);
        std::ofstream out(argv[3]);
        must(bool(cases) && bool(out), "cannot open cases or output");
        out << std::setprecision(12);
        std::string line;
        while (std::getline(cases, line)) {
            if (line.empty()) continue;
            std::stringstream row(line);
            std::string id, kind, ids_text, arg;
            must(bool(std::getline(row, id, '\t')) && bool(std::getline(row, kind, '\t')) &&
                 bool(std::getline(row, ids_text, '\t')) && bool(std::getline(row, arg, '\t')),
                 "malformed case line");
            const std::vector<int> ids = ints(ids_text, ' ');
            const int n = int(ids.size());
            must(n >= 2 && n <= n_ctx, "bad length: " + id);
            int first = 0;
            std::vector<int> cands;
            if (kind == "nll") {
                first = std::stoi(arg);
                must(first >= 1 && first < n, "bad first target: " + id);
            } else if (kind == "choice") {
                cands = ints(arg, ',');
                must(!cands.empty(), "no candidates: " + id);
                for (int c : cands) must(c >= 0 && c < vocab_size, "bad candidate: " + id);
            } else {
                throw std::runtime_error("unknown kind: " + kind);
            }
            const int count = kind == "nll" ? n - 1 : n;  // nll: the last token is only a target
            llama_memory_clear(llama_get_memory(ctx), true);
            batch.n_tokens = count;
            for (int i = 0; i < count; ++i) {
                must(ids[i] >= 0 && ids[i] < vocab_size, "bad token: " + id);
                batch.token[i] = ids[i];
                batch.pos[i] = i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = kind == "nll" ? i >= first - 1 : i == count - 1;
            }
            must(llama_decode(ctx, batch) == 0, "decode failed: " + id);
            out << "{\"id\":\"" << id << "\",\"kind\":\"" << kind << "\",\"values\":[";
            if (kind == "nll") {
                for (int t = first; t < n; ++t) {
                    const float * r = llama_get_logits_ith(ctx, t - 1);
                    must(r != nullptr, "missing logits: " + id);
                    const double v = log_normalizer(r, vocab_size) - r[ids[t]];
                    must(std::isfinite(v), "nonfinite: " + id);
                    out << (t > first ? "," : "") << v;
                }
            } else {
                const float * r = llama_get_logits_ith(ctx, count - 1);
                must(r != nullptr, "missing logits: " + id);
                const double z = log_normalizer(r, vocab_size);
                for (size_t k = 0; k < cands.size(); ++k) out << (k ? "," : "") << double(r[cands[k]]) - z;
            }
            out << "]}\n";
        }
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "scoring failed: " << e.what() << '\n';
        return 1;
    }
}
