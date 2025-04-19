#ifndef CHATGLM_H
#define CHATGLM_H

#include <string>
#include <vector>
#include "tokenizer.h"
#include "ggml.h"

class ChatGLM {
public:
    ChatGLM();
    ChatGLM(const std::string& model_path);
    ~ChatGLM();
    std::string generate(const std::string& prompt);

    // Getter and setter for model path
    std::string get_model_path() const { return model_path; }
    void load_model(const std::string& path);

    void set_tokenizer_path(const std::string& path);

private:
    std::string model_path;
    Tokenizer tokenizer;
    ggml_context *ctx = nullptr; // GGML context
};

#endif // CHATGLM_H