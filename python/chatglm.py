import chatglm

class ChatGLM:
    def __init__(self, model_path):
        self.model = chatglm.load_model(model_path)
    
    def generate(self, prompt, max_length=2048):
        return self.model.generate(prompt, max_length)


def load_model(model_path):
    return ChatGLM(model_path)