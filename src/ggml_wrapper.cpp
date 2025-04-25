#include "ggml_wrapper.h"
#include <iostream>

ggml_wrapper::ggml_wrapper(const std::string& model_path, DataType data_type) : model_path_(model_path), data_type_(data_type) {
    std::cout << "Initializing ggml with model: " << model_path_ << " and data type: " << data_type_ << std::endl;
    if (!load_model(model_path_, data_type_)) {
        throw std::runtime_error("Failed to load model.");
    }
}

ggml_wrapper::~ggml_wrapper() {
    // Free GGML context and related resources
}

std::vector<float> ggml_wrapper::forward(const std::vector<int>& input_ids) {
    // Implement the forward pass using GGML
    // This is a placeholder
    std::vector<float> logits(1000, 0.0f); // Dummy logits
    return logits;
}

bool ggml_wrapper::load_model(const std::string& model_path, DataType data_type) {
    // Load the model into GGML context based on the specified data type

    std::cout << "Loading model with data type: " << data_type << std::endl;
    //TODO: Implement GGML initialization based on data_type

    // Example implementation based on data type.  Must be adapted to the correct model loading.
    switch (data_type) {
        case FLOAT32:
            std::cout << "Using FLOAT32" << std::endl;
            //GGML load using f32
            break;
        case FLOAT16:
            std::cout << "Using FLOAT16" << std::endl;
            //GGML load using f16
            break;
        case INT8:
            std::cout << "Using INT8" << std::endl;
            //GGML load using int8 - using quantization
            break;
        default:
            std::cerr << "Unsupported data type." << std::endl;
            return false;
    }

    return true;
}
