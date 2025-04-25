#ifndef GGML_WRAPPER_H
#define GGML_WRAPPER_H

#include <string>
#include <vector>
#include "chatglm.h"

class ggml_wrapper {
public:
    ggml_wrapper(const std::string& model_path, DataType data_type = FLOAT32);
    ~ggml_wrapper();

    std::vector<float> forward(const std::vector<int>& input_ids);

private:
    std::string model_path_;
    DataType data_type_;

    // ggml context and other related variables
    void* ctx_;

    bool load_model(const std::string& model_path, DataType data_type);

};

#endif // GGML_WRAPPER_H