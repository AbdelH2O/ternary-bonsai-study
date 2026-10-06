// Pinned Prism 842b188. Validate sequence-state roundtrips before cross-checkpoint imports.
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

static void prefill(llama_context * ctx, llama_batch & batch,
                    const std::vector<llama_token> & ids, int p, int h) {
    for (int offset = 0; offset < h - 1; offset += 512) {
        const int count = std::min(512, h - 1 - offset);
        batch.n_tokens = count;
        for (int k = 0; k < count; ++k) {
            batch.token[k] = ids[p - h + offset + k];
            batch.pos[k] = offset + k;
            batch.n_seq_id[k] = 1;
            batch.seq_id[k][0] = 0;
            batch.logits[k] = false;
        }
        if (llama_decode(ctx, batch) != 0) throw std::runtime_error("prefill decode failed");
    }
    llama_synchronize(ctx);
}

static std::vector<double> score(llama_context * ctx, llama_batch & batch,
                                 const std::vector<llama_token> & ids, int p, int h, int m, int vocab_size) {
    std::vector<double> nll;
    nll.reserve(m);
    for (int offset = 0; offset < m; offset += 512) {
        const int count = std::min(512, m - offset);
        batch.n_tokens = count;
        for (int k = 0; k < count; ++k) {
            batch.token[k] = ids[p - 1 + offset + k];
            batch.pos[k] = h - 1 + offset + k;
            batch.n_seq_id[k] = 1;
            batch.seq_id[k][0] = 0;
            batch.logits[k] = true;
        }
        if (llama_decode(ctx, batch) != 0) throw std::runtime_error("continuation decode failed");
        for (int k = 0; k < count; ++k) {
            nll.push_back(gold_nll(llama_get_logits_ith(ctx, k), vocab_size, ids[p + offset + k]));
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
        if (argc != 10) throw std::runtime_error("usage: state_gate capture|transplant MODEL IDS P H M FULL_STATE PARTIAL_STATE OUTPUT_JSON");
        const std::string mode = argv[1];
        const std::string model_path = argv[2];
        const auto ids = read_ids(argv[3]);
        const int p = std::stoi(argv[4]), h = std::stoi(argv[5]), m = std::stoi(argv[6]);
        if (p < h || h < 2 || m < 1 || size_t(p + m) > ids.size() || h + m > 16384)
            throw std::runtime_error("invalid span");
        if (mode != "capture" && mode != "transplant") throw std::runtime_error("invalid mode");
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
        prefill(ctx, batch, ids, p, h);
        std::ofstream out(argv[9]);
        if (!out) throw std::runtime_error("cannot write JSON");
        out << std::setprecision(12) << "{\"mode\":\"" << mode << "\",\"model_path\":\""
            << model_path << "\",\"ids_path\":\"" << argv[3] << "\",\"p\":" << p
            << ",\"h\":" << h << ",\"m\":" << m;
        if (mode == "capture") {
            auto full = save_state(ctx, LLAMA_STATE_SEQ_FLAGS_NONE);
            auto partial = save_state(ctx, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            write_bytes(argv[7], full);
            write_bytes(argv[8], partial);
            auto direct = score(ctx, batch, ids, p, h, m, vocab_size);
            llama_memory_clear(llama_get_memory(ctx), true);
            load_state(ctx, full, LLAMA_STATE_SEQ_FLAGS_NONE);
            auto full_roundtrip = score(ctx, batch, ids, p, h, m, vocab_size);
            llama_memory_clear(llama_get_memory(ctx), true);
            load_state(ctx, full, LLAMA_STATE_SEQ_FLAGS_NONE);
            load_state(ctx, partial, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            auto partial_roundtrip = score(ctx, batch, ids, p, h, m, vocab_size);
            out << ",\"full_state_bytes\":" << full.size() << ",\"partial_state_bytes\":" << partial.size();
            out << ",\"direct_nll\":"; json_nll(out, direct);
            out << ",\"full_roundtrip_nll\":"; json_nll(out, full_roundtrip);
            out << ",\"partial_roundtrip_nll\":"; json_nll(out, partial_roundtrip);
        } else {
            // Recreate destination full state from the same history. The source partial state
            // replaces only recurrent memory, retaining the destination model's attention KV.
            auto destination_full = save_state(ctx, LLAMA_STATE_SEQ_FLAGS_NONE);
            auto direct = score(ctx, batch, ids, p, h, m, vocab_size);
            llama_memory_clear(llama_get_memory(ctx), true);
            load_state(ctx, destination_full, LLAMA_STATE_SEQ_FLAGS_NONE);
            const auto source_partial = read_bytes(argv[8]);
            load_state(ctx, source_partial, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            auto transplanted = score(ctx, batch, ids, p, h, m, vocab_size);
            out << ",\"destination_full_state_bytes\":" << destination_full.size()
                << ",\"source_partial_state_bytes\":" << source_partial.size();
            out << ",\"direct_nll\":"; json_nll(out, direct);
            out << ",\"transplanted_nll\":"; json_nll(out, transplanted);
        }
        out << "}\n";
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
