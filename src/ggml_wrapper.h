#ifndef CHATGLM_GGML_WRAPPER_H
#define CHATGLM_GGML_WRAPPER_H

#include "ggml.h"
#include <vector>
#include <stdexcept>

class GGMLTensor {
public:
    ggml_tensor *tensor;

    GGMLTensor(ggml_context *ctx, ggml_type type, const std::vector<int64_t>& dims) {
        if (dims.size() > 4) {
            throw std::runtime_error("GGML only supports up to 4 dimensions");
        }

        std::vector<int64_t> padded_dims = dims;
        while (padded_dims.size() < 4) {
            padded_dims.push_back(1);
        }

        tensor = ggml_new_tensor(ctx, type, padded_dims.data(), padded_dims.size());
        if (tensor == nullptr) {
            throw std::runtime_error("Failed to allocate GGML tensor");
        }
    }

    ~GGMLTensor() {
      // Ownership and freeing of ggml tensors will be handled by the ggml context
    }
};

#endif // CHATGLM_GGML_WRAPPER_H