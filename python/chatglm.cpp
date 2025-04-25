#include <pybind11/pybind11.h>
#include "chatglm.h"

namespace py = pybind11;

PYBIND11_MODULE(chatglmcpp, m) {
    py::enum_<DataType>(m, "DataType")
        .value("FLOAT32", FLOAT32)
        .value("FLOAT16", FLOAT16)
        .value("INT8", INT8)
        .export_values();

    py::class_<ChatGLM>(m, "ChatGLM")
        .def(py::init<const std::string&, DataType>(), py::arg("model_path"), py::arg("data_type") = FLOAT32)
        .def("generate", &ChatGLM::generate, py::arg("prompt"), py::arg("max_length") = 2048);
}