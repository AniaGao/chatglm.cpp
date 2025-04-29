import pytest
import chatglm
import os

MODEL_PATH = os.environ.get("CHATGLM_MODEL_PATH", None)
TOKENIZER_PATH = os.environ.get("CHATGLM_TOKENIZER_PATH", None)

@pytest.mark.skipif(MODEL_PATH is None or TOKENIZER_PATH is None, reason="Model and tokenizer paths must be set via environment variables CHATGLM_MODEL_PATH and CHATGLM_TOKENIZER_PATH")
class TestIntegration:
    @classmethod
    def setup_class(cls):
        cls.model = chatglm.ChatGLM(MODEL_PATH, tokenizer_path=TOKENIZER_PATH)

    def test_model_initialization(self):
        assert self.model is not None

    def test_inference(self):
        text = "你好"
        output = self.model.generate(text, max_length=32)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_batch_inference(self):
        texts = ["你好", "hello", "世界"]
        outputs = self.model.generate(texts, max_length=32)
        assert isinstance(outputs, list)
        assert len(outputs) == len(texts)
        for output in outputs:
            assert isinstance(output, str)
            assert len(output) > 0
