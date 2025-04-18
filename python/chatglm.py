# python/chatglm.py
import os
import sys
import pybind11

# Add the directory containing the compiled C++ module to the Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, module_path)

# Assuming the compiled module is named 'chatglm_cpp'
import chatglm_cpp

class ChatGLM:
    def __init__(self):
        pass

    def inference(self):
       print("Call C++ library here!")

