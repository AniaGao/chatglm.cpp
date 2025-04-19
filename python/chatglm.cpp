#include <pybind11/pybind11.h>
#include "chatglm.h"

namespace py = pybind11;

PYBIND11_MODULE(chatglm, m) {
    m.doc() = "ChatGLM C++ implementation";

    py::class_<ChatGLM>(m, "ChatGLM")
        .def(py::init<const std::string&>(), py::arg("model_path"))
        .def("generate", &ChatGLM::generate)
        .def("load_model", &ChatGLM::load_model)
        .def("set_tokenizer_path", &ChatGLM::set_tokenizer_path);
}