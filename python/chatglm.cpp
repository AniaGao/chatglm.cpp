#include <pybind11/pybind11.h>
#include "src/chatglm.cpp"

namespace py = pybind11;

PYBIND11_MODULE(chatglm_cpp, m) {
	m.doc() = "pybind11 example plugin"; // optional module docstring
	m.def("chatglm_inference", &chatglm_inference, "A function that does nothing");
}
