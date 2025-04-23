#ifndef CHATGLM_H
#define CHATGLM_H

#include <string>
#include <vector>
#include <memory>
#include "tokenizer.h"

class Model {
public:
    Model(const std::string& model_path) : model_path_(model_path) {}

    std::vector<int> generate(const std::vector<int>& tokens) const {
        // Placeholder for actual model inference logic
        // Replace with your ChatGLM model's inference implementation
        std::vector<int> generated_tokens;
        for (int i = 0; i < 10; ++i) { // Generate 10 dummy tokens
            generated_tokens.push_back(i % 10); // Example tokens
        }
        return generated_tokens;
    }

private:
    std::string model_path_;
};


class ChatGLM {
public:
    ChatGLM(const std::string& model_path, const std::string& tokenizer_path);
    std::string generate(const std::string& prompt);

private:
    std::unique_ptr<Model> model;
    Tokenizer tokenizer;

    void load_model(const std::string& model_path);
    void load_tokenizer(const std::string& tokenizer_path);

};

#endif