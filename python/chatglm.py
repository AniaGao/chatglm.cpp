import chatglm

class ChatGLM:
    def __init__(self, config):
        self.config = chatglm.Config(**config)

    def create_model_state(self):        
        return chatglm.create_model_state(self.config)

    def generate(self, state, embeddings, max_length, temperature, top_p, eos_token):
        return chatglm.generate(state, embeddings, max_length, temperature, top_p, eos_token)

    def beam_search_generate(self, initial_state, vocab_size, initial_embeddings, beam_size, max_length, eos_token):
        return chatglm.beam_search_generate(initial_state, vocab_size, initial_embeddings, beam_size, max_length, eos_token)
