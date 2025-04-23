#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include "../src/chatglm.h"

namespace py = pybind11;

PYBIND11_MODULE(chatglm, m) {
    m.doc() = "pybind11 chatglm example plugin"; // optional module docstring

    py::class_<ChatGLM>(m, "ChatGLM")
        .def(py::init<const std::string&>(), py::arg("model_path"))
        .def("generate", &ChatGLM::generate, py::arg("prompt"), py::arg("max_length") = 2048);

    m.def("load_model", [](const std::string& model_path) {
        return new ChatGLM(model_path);
    }, py::return_value_policy::take_ownership, "Load a ChatGLM model from the given path.");
}