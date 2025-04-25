#ifndef CHATGLM_CPP_BEAM_SEARCH_H
#define CHATGLM_CPP_BEAM_SEARCH_H

#include <vector>
#include <queue>
#include <algorithm>

#include "chatglm.h" // Assuming chatglm.h contains necessary definitions.

namespace chatglm {

struct BeamSearchNode {
    std::vector<int> tokens;
    float log_prob;
    ModelState state;

    bool operator<(const BeamSearchNode& other) const {
        return log_prob < other.log_prob; // For min-heap
    }
};

std::vector<std::vector<int>> beam_search(ModelState initial_state, int vocab_size, std::function<std::vector<float>(const ModelState&)> predict_fn, int beam_size, int max_len, float eos_token);

}

#endif // CHATGLM_CPP_BEAM_SEARCH_H