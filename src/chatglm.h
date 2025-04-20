#ifndef CHATGLM_H
#define CHATGLM_H

#include <vector>
#include <string>
#include <memory>

// Forward declarations
class Embedding;
class GLMBlock;
class RMSNorm;
class Linear;

class ChatGLM {
public:
    ChatGLM(int vocab_size, int hidden_size, int num_layers, int num_attention_heads);

    std::vector<float> forward(const std::vector<int>& input_ids);

private:
    int vocab_size_;
    int hidden_size_;
    int num_layers_;
    int num_attention_heads_;

    std::unique_ptr<Embedding> embedding_;
    std::vector<std::unique_ptr<GLMBlock>> layers_;
    std::unique_ptr<RMSNorm> norm_f_;
    std::unique_ptr<Linear> lm_head_;

};

class Embedding {
public:
    Embedding(int vocab_size, int hidden_size);
    std::vector<float> forward(const std::vector<int>& input_ids);
private:
    int vocab_size_;
    int hidden_size_;
    //TODO: Add weights
};

class GLMBlock {
public:
    GLMBlock(int hidden_size, int num_attention_heads);
    std::vector<float> forward(const std::vector<float>& input);

private:
    int hidden_size_;
    int num_attention_heads_;
    // TODO: Add LayerNorm, Attention, and other components
};

class RMSNorm{
public:
    RMSNorm(int hidden_size);
    std::vector<float> forward(const std::vector<float>& input);
private:
    int hidden_size_;
};

class Linear{
public:
    Linear(int in_features, int out_features);
    std::vector<float> forward(const std::vector<float>& input);
private:
    int in_features_;
    int out_features_;
};

#endif // CHATGLM_H
