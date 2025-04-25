#ifndef CHATGLM_H
#define CHATGLM_H

#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>

#include "ggml_wrapper.h"
#include "tokenizer.h"


enum DataType {
    FLOAT32,  // Full precision (float)
    FLOAT16,  // Half precision (float16)
    INT8      // 8-bit integer (quantized)
};


class ChatGLM {
public:
    ChatGLM(const std::string& model_path, DataType data_type = FLOAT32);
    ~ChatGLM();

    std::string generate(const std::string& prompt, int max_length = 2048);

private:
    std::string model_path_;
    Tokenizer tokenizer_;
    ggml_wrapper ggml_wrapper_;
    DataType data_type_;

    std::vector<float> forward(const std::vector<int>& input_ids);
    std::vector<float> softmax(const std::vector<float>& logits);
};

#endif // CHATGLM_H