import re
import torch
from datasets import load_dataset
import tqdm

class MATH500:
    """
    Loads MATH-500 via HF datasets.
    Source: HuggingFaceH4/MATH-500 (split: 'test')
    Yields dicts with: {'question': str, 'answer': str}
    """
    name = "MATH-500"

    def __init__(self, split: str = "test"):
        if load_dataset is None:
            raise RuntimeError("datasets not installed. pip install datasets")
        # Schema has fields: problem, solution, answer, subject, level, unique_id
        self.ds = load_dataset("HuggingFaceH4/MATH-500", split=split)

    def __iter__(self):
        for ex in self.ds:
            yield {"question": ex["problem"], "answer": ex["answer"]}


class MATH500_Bench:
    """
    Simple exact/normalized match evaluator for MATH-500.
    Extraction expects the model to put the final answer in \\boxed{...}.
    """
    def __init__(self, model, tokenizer, dataset, device="cuda"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device

    def generate(self, prompt, max_new_tokens=512):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,              # keep interface same as your GSM8K code
                temperature=0.0,             # greedy
                pad_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Return only the completion after the prompt
        return decoded[len(prompt):].strip()

    # ---------- Answer parsing / normalization helpers ----------

    @staticmethod
    def _extract_boxed(text: str):
        """
        Get the LAST \\boxed{...} occurrence (common convention in math evals).
        """
        matches = re.findall(r"\\boxed\\{(.+?)\\}", text, flags=re.DOTALL)
        return matches[-1].strip() if matches else None

    @staticmethod
    def _fallback_tail(text: str):
        """
        Fallback: use the last non-empty line as a crude final answer.
        """
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        return lines[-1] if lines else None

    @staticmethod
    def _latex_normalize(s: str):
        """
        Light normalization for LaTeX-ish strings before exact compare.
        (Removes whitespace, \\left/\\right, surrounding $...$, etc.)
        """
        if s is None:
            return None
        s = s.strip()
        # strip surrounding $...$
        if s.startswith("$") and s.endswith("$"):
            s = s[1:-1].strip()
        # remove \left, \right and thin spaces
        s = re.sub(r"\\left|\\right|\\,", "", s)
        # collapse whitespace
        s = re.sub(r"\s+", "", s)
        return s

    def _extract_pred_answer(self, text: str):
        """
        Try \\boxed{...} first, then fallback to last line.
        """
        boxed = self._extract_boxed(text)
        if boxed:
            return boxed
        return self._fallback_tail(text)

    # ---------- Evaluation ----------

    def evaluate(self, limit=None, prompt: str = None):
        """
        Returns:
            {
              "accuracy": float,
              "total": int,
              "correct": int,
              "details": [ ... per-example dicts ... ]
            }
        """
        total, correct = 0, 0
        results = []

        if prompt is None:
            prompt = (
                "Solve the following problem step by step. "
                "Provide ONLY the final answer wrapped as \\boxed{...}.\n\n"
                "Problem: {query}\nAnswer:"
            )

        pbar = tqdm(self.dataset, desc=f"Evaluating MATH-500: acc - {correct}/{total}")
        for i, ex in enumerate(pbar):
            if limit and i >= limit:
                break

            q, gold = ex["question"], ex["answer"]

            pred_text = self.generate(prompt.format(query=q), max_new_tokens=512)
            pred_ans_raw = self._extract_pred_answer(pred_text)

            gold_n = self._latex_normalize(gold)
            pred_n = self._latex_normalize(pred_ans_raw)

            is_correct = (pred_n == gold_n)
            total += 1
            correct += int(is_correct)

            results.append({
                "bench": 'MATH-500',
                "question": q,
                "gold": gold,                  # keep original gold for inspection
                "pred_text": pred_text,        # full generation
                "pred_num": pred_ans_raw,    # extracted final answer
                # "match_norm": (pred_n, gold_n),
                "correct": is_correct,
            })

            acc = correct / total if total > 0 else 0.0
            pbar.set_description(f"Evaluating MATH-500: acc - {correct}/{total} ({acc:.2%})")
        pbar.close()

        return {"accuracy": acc, "total": total, "correct": correct, "details": results}