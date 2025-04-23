import chatglm

class Model:
    def __init__(self, model_path, tokenizer_path):
        self.model = chatglm.ChatGLM(model_path, tokenizer_path)

    def generate(self, prompt):
        return self.model.generate(prompt)


if __name__ == '__main__':
    # Example usage (replace with actual paths)
    model_path = "path/to/your/model"  # Replace with your model path
    tokenizer_path = "path/to/your/tokenizer.model"  # Replace with your tokenizer path
    model = Model(model_path, tokenizer_path)
    prompt = "Hello, how are you?"
    response = model.generate(prompt)
    print(f"Prompt: {prompt}\nResponse: {response}")