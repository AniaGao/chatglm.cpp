#ifndef CHATGLM_H
#define CHATGLM_H

#include <vector>
#include <string>
#include <iostream>

class ChatGLM {
public:
    ChatGLM();
    ~ChatGLM();

    bool load_model(const std::string& filename);
    std::vector<float> forward(const std::vector<float>& input);

private:
    // Model parameters (example)
    int num_layers;
    int hidden_size;
    std::vector<std::vector<float>> weights; // Simplified weight storage

    bool initialized = false;
};

#endif