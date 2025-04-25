#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "chatglm.h"
#include "tokenizer.h"

namespace py = pybind11;

PYBIND11_MODULE(chatglm, m) {
    m.doc() = "ChatGLM C++ implementation";

    py::class_<ChatGLM>(m, "ChatGLM")
        .def(py::init<const std::string&>(), py::arg("model_path"))
        .def("generate", &ChatGLM::generate, py::arg("prompt"), py::arg("max_length") = 2048);

    py::class_<tokenizer::Tokenizer>(m, "Tokenizer")
        .def(py::init<const std::string&>(), py::arg("model_path"))
        .def("encode", &tokenizer::Tokenizer::Encode, py::arg("text"))
        .def("decode", &tokenizer::Tokenizer::Decode, py::arg("tokens"));
}