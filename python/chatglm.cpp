#include <pybind11/pybind11.h>
#include "../src/chatglm.h"

namespace py = pybind11;

PYBIND11_MODULE(chatglm, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    py::class_<ChatGLM>(m, "ChatGLM")
        .def(py::init<const std::string&>(), py::arg("model_path"))
        .def("generate", &ChatGLM::generate, py::arg("prompt"), py::arg("max_length") = 2048);

    //Example function, remove after actual implementation
    m.def("test_func", []() { return "Hello from C++"; }, "A test function");
}