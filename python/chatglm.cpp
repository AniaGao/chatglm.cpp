#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../src/chatglm.h"

namespace py = pybind11;

PYBIND11_MODULE(chatglm, m) {
    py::class_<ChatGLM>(m, "ChatGLM")
        .def(py::init<const std::string&, const std::string&>())
        .def("generate", &ChatGLM::generate, "Generate text from a prompt");
}