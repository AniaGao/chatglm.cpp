#include <pybind11/pybind11.h>
#include "src/chatglm.h"
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(chatglm, m) {
    py::class_<ChatGLM>(m, "ChatGLM")
        .def(py::init<>())
        .def("load_model", &ChatGLM::load_model)
        .def("forward", [](ChatGLM &model, const py::list &input_list) {
            std::vector<float> input;
            for (auto item : input_list) {
                input.push_back(py::cast<float>(item));
            }
            std::vector<float> output = model.forward(input);
            py::list output_list;
            for (float val : output) {
                output_list.append(val);
            }
            return output_list;
        });
}