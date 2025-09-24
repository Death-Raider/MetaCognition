import torch
import re
from datasets import load_dataset
from tqdm import tqdm

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
    def __len__(self):
        return len(self.ds)
    def __iter__(self):
        for ex in self.ds:
            yield {"question": ex["question"], "answer": ex["answer"]}
    
class GSM8K_Bench:
    def __init__(self, model, tokenizer, dataset, device="cuda"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device

    def generate_batch(self, prompts, max_new_tokens=256):
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side='left'
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode all outputs
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Slice off the original prompt text
        results = []
        for prompt, text in zip(prompts, decoded):
            results.append(text[len(prompt):].strip())
        return results

    @staticmethod
    def extract_number(text):
        """
        Extract the final numeric answer from GSM8K model output.
        GSM8K gold answers are in the form '#### <number>'
        """
        numbers = re.findall(r"-?\d+\.?\d*", text)
        return numbers[-1] if numbers else None

    def evaluate(self, limit=None, prompt: str = None, batch_size: int = 8):
        total, correct = 0, 0
        results = []

        if prompt is None:
            prompt = (
                "Solve the following math problem step-by-step. "
                "Let's think step by step.\n\n"
                "Q: {query}\nA:"
            )

        # Create an iterator over the dataset
        iterator = iter(self.dataset)

        if limit:
            total_len = limit
        else:
            total_len = len(self.dataset)

        pbar = tqdm(range(0, total_len, batch_size),
                    desc=f"Evaluating GSM8K: acc - {correct}/{total}")

        for _ in pbar:
            # Collect a batch of examples
            batch = []
            for _ in range(batch_size):
                try:
                    ex = next(iterator)
                    batch.append(ex)
                except StopIteration:
                    break
            if not batch:
                break

            # Format prompts and gold answers
            prompts = [prompt.format(query=ex["question"]) for ex in batch]
            golds = [ex["answer"].split("####")[-1].strip() for ex in batch]

            preds_text = self.generate_batch(prompts, max_new_tokens=512)

            for ex, gold_num, pred_text in zip(batch, golds, preds_text):
                pred_num = self.extract_number(pred_text)
                is_correct = (pred_num == gold_num)

                total += 1
                correct += int(is_correct)

                results.append({
                    "bench": "GSM8K",
                    "question": ex["question"],
                    "gold": gold_num,
                    "pred_text": pred_text,
                    "pred_num": pred_num,
                    "correct": is_correct,
                })

            acc = correct / total if total > 0 else 0.0
            pbar.set_description(f"Evaluating GSM8K: acc - {correct}/{total} ({acc:.2%})")

            if limit and total >= limit:
                break

        pbar.close()
        return {"accuracy": acc, "total": total, "correct": correct, "details": results}