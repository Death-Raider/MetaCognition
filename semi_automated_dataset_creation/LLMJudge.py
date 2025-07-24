import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

class Judge:
    def __init__(self):
        self.model_name = ""
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_len = 512
    
    def load_model(self, model_name, tokenizer_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def judge(self, input_text):
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=self.max_len).to(self.device)
        outputs = self.model.generate(**inputs, max_length=self.max_len, num_return_sequences=1)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
    