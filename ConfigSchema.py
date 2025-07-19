import json
class ConfigSchema:
    def __init__(self, model_name: str='', max_len: int = 0, lr: float = 0.0, batch_size: int = 0, epochs: int = 0, beta: float = 0.0):
        self.model_name = model_name
        if self.model_name != '':
            assert self.model_name in self.allowed_models()
        self.max_len = max_len
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.beta = beta

    def __str__(self):
        return json.dumps(self.to_dict())
    
    def to_dict(self):
        return {
            "model_name": self.model_name,
            "max_len": self.max_len,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "beta": self.beta
        }
    
    def from_dict(self, dict):
        self.model_name = str(dict.get("MODEL_NAME", self.model_name))
        assert self.model_name in self.allowed_models()
        self.max_len = int(dict.get("MAX_LEN", self.max_len))
        self.lr = float(dict.get("LR", self.lr))
        self.batch_size = int(dict.get("BATCH_SIZE", self.batch_size))
        self.epochs = int(dict.get("EPOCHS", self.epochs))
        self.beta = float(dict.get("BETA", self.beta))

    def allowed_models(self):
        # https://huggingface.co/collections/ehristoforu/the-best-small-llm-instruct-models-669e89c6263d01888798cb7a
        SMOL_MODELS = [
            'microsoft/phi-2',
            'Qwen/Qwen1.5-1.8B',
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        ]
        BIGG_MODELS = [
            'Qwen/Qwen1.5-7B', # it just a hater for no reason
            'mistralai/Mistral-7B-Instruct-v0.3'
        ]
        return SMOL_MODELS + BIGG_MODELS