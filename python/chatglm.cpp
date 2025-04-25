#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "src/chatglm.h"
#include "src/utils.h"
#include "src/tokenizer.h"

namespace py = pybind11;

PYBIND11_MODULE(chatglm, m) {
    m.doc() = "chatglm.cpp python bindings";

    py::class_<chatglm::Config>(m, "Config")
        .def(py::init<int, int, int, int, int, int, int, int, int, float, int, int, int, int, int>())
        .def_readwrite("embedding_size", &chatglm::Config::embedding_size)
        .def_readwrite("num_layers", &chatglm::Config::num_layers)
        .def_readwrite("num_attention_heads", &chatglm::Config::num_attention_heads)
        .def_readwrite("ffn_hidden_size", &chatglm::Config::ffn_hidden_size)
        .def_readwrite("kv_channels", &chatglm::Config::kv_channels)
        .def_readwrite("vocab_size", &chatglm::Config::vocab_size)
        .def_readwrite("padded_vocab_size", &chatglm::Config::padded_vocab_size)
        .def_readwrite("seq_length", &chatglm::Config::seq_length)
        .def_readwrite("num_kv_heads", &chatglm::Config::num_kv_heads)
        .def_readwrite("layer_norm_epsilon", &chatglm::Config::layer_norm_epsilon)
        .def_readwrite("rope_theta", &chatglm::Config::rope_theta)
        .def_readwrite("use_dynamic_ntk", &chatglm::Config::use_dynamic_ntk)
        .def_readwrite("use_logn_attn", &chatglm::Config::use_logn_attn)
        .def_readwrite("use_flash_attn", &chatglm::Config::use_flash_attn)
        .def_readwrite("num_devices", &chatglm::Config::num_devices);

    py::class_<chatglm::ModelState>(m, "ModelState");

    m.def("create_model_state", &chatglm::create_model_state);
    m.def("model_forward", &chatglm::model_forward);
    m.def("generate", &chatglm::generate);

    m.def("beam_search_generate", &chatglm::beam_search_generate, 
          py::arg("initial_state"),
          py::arg("vocab_size"),
          py::arg("initial_embeddings"),
          py::arg("beam_size"),
          py::arg("max_length"),
          py::arg("eos_token"),
          "Generate sequences using beam search.");
}
