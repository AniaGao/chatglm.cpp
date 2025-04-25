#include "chatglm.h"
#include "utils.h"
#include "tokenizer.h"
#include "ggml_wrapper.h"
#include "sentencepiece.h"
#include "beam_search.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cassert>

namespace chatglm {

ModelState create_model_state(const Config& config, ggml_context* ctx) {
    ModelState state;

    state.config = config;
    state.ctx = ctx;

    state.x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.padded_vocab_size);

    state.transformer = create_transformer_state(state.config, state.ctx);

    return state;
}

std::vector<float> model_forward(ModelState& state, const std::vector<float>& embeddings) {
    // Copy embeddings to input tensor
    ggml_tensor* x = state.x;
    memcpy(x->data, embeddings.data(), embeddings.size() * ggml_type_sizef(x->type));

    // Transformer forward
    transformer_forward(state.transformer, x);

    // Calculate logits
    ggml_tensor* logits = ggml_norm(state.ctx, state.transformer.output);
    logits = ggml_mul_mat(state.ctx, state.transformer.output, state.transformer.wcls);
    logits = ggml_add(state.ctx, state.transformer.bcls, logits);

    ggml_tensor* probs = ggml_soft_max(state.ctx, logits);

    ggml_build_forward_expand(&state.gf, probs);
    ggml_graph_compute(state.gf);

    std::vector<float> final_probs(state.config.padded_vocab_size);
    memcpy(final_probs.data(), probs->data, state.config.padded_vocab_size * ggml_type_sizef(probs->type));

    ggml_free(state.gf);

    return final_probs;
}

int sample(const std::vector<float>& logits, float temperature, float top_p) {
    std::vector<float> probs = logits;
    // Apply temperature
    for (auto& prob : probs) {
        prob = exp(prob / temperature);
    }

    // Normalize
    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    for (auto& prob : probs) {
        prob /= sum;
    }

    // Top-p sampling
    std::vector<std::pair<float, int>> prob_index;
    for (int i = 0; i < probs.size(); ++i) {
        prob_index.emplace_back(probs[i], i);
    }
    std::sort(prob_index.begin(), prob_index.end(), std::greater<std::pair<float, int>>());

    float cumulative_prob = 0.0f;
    std::vector<std::pair<float, int>> top_p_indices;
    for (const auto& pi : prob_index) {
        top_p_indices.push_back(pi);
        cumulative_prob += pi.first;
        if (cumulative_prob > top_p) {
            break;
        }
    }

    // Normalize top-p probabilities
    sum = 0.0f;
    for (const auto& pi : top_p_indices) {
        sum += pi.first;
    }
    for (auto& pi : top_p_indices) {
        pi.first /= sum;
    }

    // Sample from top-p distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(top_p_indices.size(), 0.0, top_p_indices.size(), [&](double x) { return top_p_indices[(int)x].first; });
    int index = dist(gen);
    return top_p_indices[index].second;
}


std::vector<int> generate(ModelState& state, const std::vector<float>& embeddings, int max_length, float temperature, float top_p, int eos_token) {
    std::vector<int> tokens;
    std::vector<float> current_embeddings = embeddings;
    for (int i = 0; i < max_length; ++i) {
        std::vector<float> logits = model_forward(state, current_embeddings);
        int next_token = sample(logits, temperature, top_p);
        tokens.push_back(next_token);

        if (next_token == eos_token) {
            break;
        }

        // Create new embeddings for the next token.  In a real implemenation, this would likely use a learned embedding table.
        current_embeddings.resize(state.config.embedding_size);
        std::fill(current_embeddings.begin(), current_embeddings.end(), 0.0f);
        current_embeddings[0] = next_token; // Use the token ID as a simple embedding for demonstration.
    }
    return tokens;
}


std::vector<std::vector<int>> beam_search_generate(ModelState initial_state, int vocab_size, const std::vector<float>& initial_embeddings, int beam_size, int max_length, float eos_token) {
  // Define a lambda function to wrap the model_forward call
  auto predict_fn = [&](const ModelState& state) {
    // Create a copy to avoid modifying the original state passed to beam_search
    ModelState mutable_state = state;
    return model_forward(mutable_state, initial_embeddings); //initial embeddings passed to first token only
  };

  return beam_search(initial_state, vocab_size, predict_fn, beam_size, max_length, eos_token);
}


}


namespace chatglm {

std::vector<std::vector<int>> beam_search(ModelState initial_state, int vocab_size, std::function<std::vector<float>(const ModelState&)> predict_fn, int beam_size, int max_len, float eos_token) {
    std::priority_queue<BeamSearchNode> beam;
    beam.push({{}, 0.0f, initial_state});

    std::vector<std::vector<int>> completed_sequences;

    while (!beam.empty()) {
        BeamSearchNode current = beam.top();
        beam.pop();

        if (current.tokens.size() >= max_len || (current.tokens.size() > 0 && current.tokens.back() == eos_token)) {
            completed_sequences.push_back(current.tokens);
            if (completed_sequences.size() >= beam_size) {
                break; // Stop if we have enough completed sequences
            }
            continue;
        }

        std::vector<float> logits = predict_fn(current.state);
        std::vector<std::pair<float, int>> ranked_logits(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
          ranked_logits[i] = {logits[i], i};
        }

        std::sort(ranked_logits.begin(), ranked_logits.end(), std::greater<std::pair<float, int>>());

        for (int i = 0; i < beam_size; ++i) {
          int next_token = ranked_logits[i].second;
          float log_prob = ranked_logits[i].first;

            // Create new embeddings for the next token.  In a real implemenation, this would likely use a learned embedding table.
            std::vector<float> next_embeddings(initial_state.config.embedding_size);
            std::fill(next_embeddings.begin(), next_embeddings.end(), 0.0f);
            next_embeddings[0] = next_token; // Use the token ID as a simple embedding for demonstration.

            ModelState next_state = current.state; // Copy the current state

            // Run one forward pass to update the model state.
            model_forward(next_state, next_embeddings);

          std::vector<int> next_tokens = current.tokens;
          next_tokens.push_back(next_token);

          beam.push({next_tokens, current.log_prob + log_prob, next_state});
        }
    }

        // Sort completed sequences by log probability
        std::sort(completed_sequences.begin(), completed_sequences.end(), [&](const std::vector<int>& a, const std::vector<int>& b) {
            float log_prob_a = 0.0f;
            float log_prob_b = 0.0f;

            // Dummy log_prob calculations
            log_prob_a = a.size();
            log_prob_b = b.size();

            return log_prob_a > log_prob_b;
        });

    return completed_sequences;
}

} // namespace chatglm