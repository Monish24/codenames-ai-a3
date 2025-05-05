from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class LocalLLM:
    def __init__(self, model_name = "declare-lab/flan-alpaca-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.model.eval()

    def ask(self, prompt, max_new_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
