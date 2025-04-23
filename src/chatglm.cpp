#include "chatglm.h"
#include "utils.h"
#include <iostream>
#include <vector>

std::string ChatGLM::generate(const std::string& prompt) {
    // Tokenize the prompt
    std::vector<int> tokens = tokenizer.encode(prompt);

    // Perform inference (simplified for single prompt)
    std::vector<int> output_tokens = model->generate(tokens);

    // Detokenize the output
    std::string output = tokenizer.decode(output_tokens);

    return output;
}

void ChatGLM::load_model(const std::string& model_path) {
  model = std::make_unique<Model>(model_path);
}

void ChatGLM::load_tokenizer(const std::string& tokenizer_path) {
  tokenizer = Tokenizer(tokenizer_path);
}

ChatGLM::ChatGLM(const std::string& model_path, const std::string& tokenizer_path) {
  load_model(model_path);
  load_tokenizer(tokenizer_path);
}