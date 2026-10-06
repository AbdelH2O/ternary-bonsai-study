// Pinned Prism 842b188. Controlled full and recurrent-only sequence-state comparisons.
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<llama_token> read_ids(const std::string & path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open IDs: " + path);
    std::vector<llama_token> ids;
    llama_token id;
    while (in >> id) ids.push_back(id);
    if (!in.eof()) throw std::runtime_error("invalid IDs: " + path);
    return ids;
}

static std::vector<uint8_t> read_bytes(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open state: " + path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

static void write_bytes(const std::string & path, const std::vector<uint8_t> & data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write state: " + path);
    out.write(reinterpret_cast<const char *>(data.data()), data.size());
    if (!out) throw std::runtime_error("state write failed: " + path);
}

static std::vector<uint8_t> save_state(llama_context * ctx, llama_state_seq_flags flags) {
    const size_t capacity = llama_state_seq_get_size_ext(ctx, 0, flags);
    if (!capacity) throw std::runtime_error("state size is zero");
    std::vector<uint8_t> data(capacity);
    const size_t written = llama_state_seq_get_data_ext(ctx, data.data(), data.size(), 0, flags);
    if (!written || written > capacity) throw std::runtime_error("state serialization failed");
    data.resize(written);
    return data;
}

static void load_state(llama_context * ctx, const std::vector<uint8_t> & data, llama_state_seq_flags flags) {
    const size_t read = llama_state_seq_set_data_ext(ctx, data.data(), data.size(), 0, flags);
    if (read != data.size()) throw std::runtime_error("state load consumed wrong byte count: " + std::to_string(read));
}

static double gold_nll(const float * logits, int vocab_size, llama_token gold) {
    if (!logits || gold < 0 || gold >= vocab_size) throw std::runtime_error("invalid gold logit");
    float max_logit = logits[0];
    for (int j = 1; j < vocab_size; ++j) max_logit = std::max(max_logit, logits[j]);
    double sum = 0;
    for (int j = 0; j < vocab_size; ++j) sum += std::exp(double(logits[j] - max_logit));
    return double(max_logit) + std::log(sum) - double(logits[gold]);
}

static double prefill(llama_context * ctx, llama_batch & batch,
                      const std::vector<llama_token> & ids, int p, int h, int vocab_size) {
    double first_nll = -1;
    for (int offset = 0; offset < h; offset += 512) {
        const int count = std::min(512, h - offset);
        batch.n_tokens = count;
        for (int k = 0; k < count; ++k) {
            batch.token[k] = ids[p - h + offset + k];
            batch.pos[k] = offset + k;
            batch.n_seq_id[k] = 1;
            batch.seq_id[k][0] = 0;
            batch.logits[k] = offset + k == h - 1;
        }
        if (llama_decode(ctx, batch) != 0) throw std::runtime_error("prefill decode failed");
        if (offset + count == h) {
            first_nll = gold_nll(llama_get_logits_ith(ctx, count - 1), vocab_size, ids[p]);
        }
    }
    llama_synchronize(ctx);
    if (first_nll < 0) throw std::runtime_error("first target logit missing");
    return first_nll;
}

static std::vector<double> score(llama_context * ctx, llama_batch & batch,
                                 const std::vector<llama_token> & ids, int p, int h, int m,
                                 int vocab_size, double first_nll) {
    std::vector<double> nll;
    nll.reserve(m);
    nll.push_back(first_nll);
    for (int offset = 0; offset < m - 1; offset += 512) {
        const int count = std::min(512, m - 1 - offset);
        batch.n_tokens = count;
        for (int k = 0; k < count; ++k) {
            batch.token[k] = ids[p + offset + k];
            batch.pos[k] = h + offset + k;
            batch.n_seq_id[k] = 1;
            batch.seq_id[k][0] = 0;
            batch.logits[k] = true;
        }
        if (llama_decode(ctx, batch) != 0) throw std::runtime_error("continuation decode failed");
        for (int k = 0; k < count; ++k) {
            nll.push_back(gold_nll(llama_get_logits_ith(ctx, k), vocab_size, ids[p + offset + k + 1]));
        }
    }
    llama_synchronize(ctx);
    return nll;
}

static void json_nll(std::ostream & out, const std::vector<double> & values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << values[i];
    out << ']';
}

int main(int argc, char ** argv) {
    try {
        if (argc < 10 || (argc - 7) % 3 != 0) throw std::runtime_error(
            "usage: controlled_state MODEL IDS P H M OUTPUT_JSON LABEL partial|full STATE_FILE [...]");
        const std::string model_path = argv[1];
        const auto ids = read_ids(argv[2]);
        const int p = std::stoi(argv[3]), h = std::stoi(argv[4]), m = std::stoi(argv[5]);
        if (p < h || h < 2 || m < 1 || size_t(p + m) > ids.size() || h + m > 16384)
            throw std::runtime_error("invalid span");
        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 99;
        llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
        if (!model) throw std::runtime_error("model load failed");
        auto params = llama_context_default_params();
        params.n_ctx = 16384;
        params.n_seq_max = 1;
        params.n_batch = 512;
        params.n_ubatch = 512;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        llama_context * ctx = llama_init_from_model(model, params);
        if (!ctx) throw std::runtime_error("context creation failed");
        llama_batch batch = llama_batch_init(512, 0, 1);
        const int vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const double first_nll = prefill(ctx, batch, ids, p, h, vocab_size);
        const auto destination_full = save_state(ctx, LLAMA_STATE_SEQ_FLAGS_NONE);
        const auto direct = score(ctx, batch, ids, p, h, m, vocab_size, first_nll);
        std::ofstream out(argv[6]);
        if (!out) throw std::runtime_error("cannot write JSON");
        out << std::setprecision(12) << "{\"model_path\":\"" << model_path
            << "\",\"ids_path\":\"" << argv[2] << "\",\"p\":" << p
            << ",\"h\":" << h << ",\"m\":" << m
            << ",\"destination_full_state_bytes\":" << destination_full.size()
            << ",\"direct_nll\":";
        json_nll(out, direct);
        out << ",\"conditions\":[";
        for (int arg = 7; arg < argc; arg += 3) {
            const std::string label = argv[arg];
            const std::string kind = argv[arg + 1];
            const std::string state_path = argv[arg + 2];
            if (kind != "partial" && kind != "full") throw std::runtime_error("invalid state kind: " + kind);
            const auto source = read_bytes(state_path);
            llama_memory_clear(llama_get_memory(ctx), true);
            if (kind == "partial") {
                load_state(ctx, destination_full, LLAMA_STATE_SEQ_FLAGS_NONE);
                load_state(ctx, source, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            } else {
                load_state(ctx, source, LLAMA_STATE_SEQ_FLAGS_NONE);
            }
            const auto imported = score(ctx, batch, ids, p, h, m, vocab_size, first_nll);
            llama_memory_clear(llama_get_memory(ctx), true);
            load_state(ctx, destination_full, LLAMA_STATE_SEQ_FLAGS_NONE);
            const auto restored = score(ctx, batch, ids, p, h, m, vocab_size, first_nll);
            out << (arg == 7 ? "" : ",") << "{\"label\":\"" << label
                << "\",\"kind\":\"" << kind << "\",\"state_path\":\""
                << state_path << "\",\"state_bytes\":" << source.size()
                << ",\"imported_nll\":";
            json_nll(out, imported);
            out << ",\"restored_nll\":";
            json_nll(out, restored);
            out << "}";
            out.flush();
        }
        out << "]}\n";
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
