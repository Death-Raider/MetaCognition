import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
class Judge:
    def __init__(self):
        self.model_name = ""
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_len = 512
    