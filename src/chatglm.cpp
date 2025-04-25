#include "chatglm.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <numeric>

ChatGLM::ChatGLM(const std::string& model_path, DataType data_type) : model_path_(model_path), tokenizer_(model_path), ggml_wrapper_(model_path, data_type), data_type_(data_type) {
    std::cout << "Loading model from: " << model_path_ << std::endl;
    // Load the model here using ggml
    // For simplicity, assume loading within ggml_wrapper constructor
}

ChatGLM::~ChatGLM() {
    // Free resources
}

std::string ChatGLM::generate(const std::string& prompt, int max_length) {
    std::vector<int> input_ids = tokenizer_.encode(prompt);

    std::string result = prompt;
    for (int i = 0; i < max_length; ++i) {
        std::vector<float> logits = forward(input_ids);
        std::vector<float> probs = softmax(logits);

        // Sample the next token (simple argmax for now)
        int next_token_id = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));

        if (next_token_id == tokenizer_.eos_token_id()) {
            break;
        }

        result += tokenizer_.decode({next_token_id});
        input_ids.push_back(next_token_id);
    }

    return result;
}

std::vector<float> ChatGLM::forward(const std::vector<int>& input_ids) {
    // Perform the forward pass using ggml
    return ggml_wrapper_.forward(input_ids);
}

std::vector<float> ChatGLM::softmax(const std::vector<float>& logits) {
    std::vector<float> result(logits.size());
    float max_val = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        result[i] = exp(logits[i] - max_val);
        sum += result[i];
    }
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] /= sum;
    }
    return result;
}
