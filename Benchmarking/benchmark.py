from datasets import load_dataset
import torch
from tqdm import tqdm
import re
from logger import logger

class GSM8K:
    """
    Loads GSM8K (main split) via HF datasets.
    Yields dicts with: {'question': str, 'answer': str}
    """
    name = "GSM8K"

    def __init__(self, split: str = "test", subset: str = "main"):
        if load_dataset is None:
            raise RuntimeError("datasets not installed. pip install datasets")
        self.ds = load_dataset("gsm8k", subset, split=split)

    def __iter__(self):
        for ex in self.ds:
            yield {"question": ex["question"], "answer": ex["answer"]}


class GSM8K_Bench:
    def __init__(self, model, tokenizer, dataset, device="cuda"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device

    def generate(self, prompt, max_new_tokens=256):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded[len(prompt):].strip()

    @staticmethod
    def extract_number(text):
        """
        Extract the final numeric answer from GSM8K model output.
        GSM8K gold answers are in the form '#### <number>'
        """
        # Prefer the last number in the string
        numbers = re.findall(r"-?\d+\.?\d*", text)
        return numbers[-1] if numbers else None

    def evaluate(self, limit=None, prompt=None):
        total, correct = 0, 0
        results = []
        if prompt is None:
            prompt = (
                "Solve the following math problem step-by-step. "
                "Let's think step by step.\n\n"
                "Q: {query}\nA:"
            )
        for i, ex in enumerate(tqdm(self.dataset, desc=f"Evaluating GSM8K: acc - {correct}/{total}")):
            if limit and i >= limit:
                break

            q, gold = ex["question"], ex["answer"]

            # Gold answers are like "#### 42"
            gold_num = gold.split("####")[-1].strip()

            pred_text = self.generate(prompt.format(query=q), max_new_tokens=512)

            pred_num = self.extract_number(pred_text)

            is_correct = (pred_num == gold_num)
            total += 1
            correct += int(is_correct)

            results.append({
                "question": q,
                "gold": gold_num,
                "pred_text": pred_text,
                "pred_num": pred_num,
                "correct": is_correct,
            })
            logger.info(f"Q: {q}\nG: {gold_num}\nP: {pred_text}\nCorrect: {is_correct}\n")

        acc = correct / total if total > 0 else 0.0
        return {"accuracy": acc, "total": total, "correct": correct, "details": results}


def bench(model, tokenizer, prompt_instruction=None):
    gsm8k = GSM8K()
    bench = GSM8K_Bench(model, tokenizer, gsm8k, device='auto', prompt=prompt_instruction)
    results = bench.evaluate(limit=50, prompt=prompt_instruction)  # limit for quicker test run
    print(f"GSM8K Accuracy: {results['accuracy']*100:.2f}% "
          f"({results['correct']}/{results['total']})")
