from typing import Optional
import chatglmcpp

class ChatGLM:
    def __init__(self, model_path: str, data_type: str = "float32") -> None:
        self.model = chatglmcpp.ChatGLM(model_path, self._convert_data_type(data_type))

    def generate(self, prompt: str, max_length: int = 2048) -> str:
        return self.model.generate(prompt, max_length)

    def _convert_data_type(self, data_type: str) -> int:
        if data_type.lower() == "float32":
            return 0  # Assuming 0 maps to FLOAT32 in C++
        elif data_type.lower() == "float16":
            return 1  # Assuming 1 maps to FLOAT16 in C++
        elif data_type.lower() == "int8":
            return 2  # Assuming 2 maps to INT8 in C++
        else:
            raise ValueError("Invalid data type.  Must be float32, float16, or int8")
