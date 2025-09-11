import re
import torch
from datasets import load_dataset
from tqdm import tqdm

try:
    import sympy as sp
    HAVE_SYMPY = True
except Exception:
    HAVE_SYMPY = False

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
                temperature=0.1,             # greedy
                pad_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Return only the completion after the prompt
        return decoded[len(prompt):].strip()

    # ---------- Answer parsing / normalization helpers ----------

    @staticmethod
    def _extract_boxed(text: str):
            """
            Prefer the last occurrence of \boxed{...}.
            Accept optional surrounding $ ... $ or \( ... \).
            """
            if not text:
                return None
            # 1) match optional $ or \( around \boxed{...}
            boxed_matches = re.findall(r'\\boxed\{((?:[^{}]|{[^{}]*})*)\}', text, flags=re.DOTALL)
            if boxed_matches:
                return boxed_matches
        
            return None

    @staticmethod
    def _fallback_last_math_line(text: str):
        """
        If no boxed answer, try to use the last non-empty line that looks like math.
        We try a few heuristics: contains digits, parentheses, \frac, or typical math tokens.
        """
        if not text:
            return None
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None
        # check last few lines for math-like content
        for ln in reversed(lines[-3:]):  # examine up to last 3 non-empty lines
            marked = re.findall(r"[0-9\\\(\)\[\]\{\}\+\-\*/=]|\\frac|\\pi|sqrt", ln)
            if marked is not None and len(marked) >= 1:
                return marked
        # else just return the last non-empty line
        return [lines[-1].strip()]

    @staticmethod
    def _latex_normalize(s: str):
        """
        Lightweight normalization of LaTeX-ish final answers to a canonical string.
        - strips wrapping $...$ or \( ... \)
        - removes \left/\right and thin spaces
        - converts simple \frac{a}{b} -> (a)/(b)
        - collapses whitespace
        Returns normalized string (or None).
        """
        if s is None:
            return None
        s = s.strip()
        # strip surrounding $ or \( \) or \( ... \) occurrences
        s = re.sub(r"^\$+|\\begin\{.*?\}|\\end\{.*?\}|\$+$", "", s)
        s = s.strip()
        # remove surrounding \( \) or \[ \]
        s = re.sub(r"^\\\(|\\\)$", "", s)
        s = s.strip()
        # remove \left and \right
        s = re.sub(r"\\left|\\right", "", s)
        # convert \pi to "pi" (so both sides are consistent)
        s = s.replace(r"\pi", "pi")
        # convert \frac{a}{b} to (a)/(b)
        s = re.sub(r"\\frac\{\s*([^\{\}]+?)\s*\}\{\s*([^\{\}]+?)\s*\}", r"(\1)/(\2)", s)
        # remove redundant whitespace
        s = re.sub(r"\s+", "", s)
        return s
        
    @staticmethod
    def _canonicalize(s: str):
        """
        Try to create a canonical representation. If sympy is available, attempt symbolic simplification.
        Otherwise return the normalized string.
        """
        if s is None:
            return None
        norm = MATH500_Bench._latex_normalize(s)
        if norm is None:
            return None
    
        if HAVE_SYMPY:
            try:
                # Replace caret with ** for pow if present
                norm_for_sympy = norm.replace("^", "**")
                expr = sp.sympify(norm_for_sympy)
                simplified = sp.simplify(expr)
                return simplified  # sympy object
            except Exception:
                # fallback to normalized string
                return norm
        else:
            return norm
    def _extract_pred_answer(self, text: str):
        """
        Try \\boxed{...} first, then fallback to last line.
        """
        boxed = self._extract_boxed(text)
        if boxed:
            return boxed
        return self._fallback_last_math_line(text)

    # ---------- Evaluation ----------

    def sympy_checker(self, expr1, expr2):
        is_correct = False
        if HAVE_SYMPY and isinstance(expr1, sp.Expr) and isinstance(expr2, sp.Expr):
            try:
                is_correct = (sp.simplify(expr1 - expr2) == 0) 
            except Exception:
                is_correct = (str(expr1) == str(expr2))
        else:
            is_correct = (str(expr1) == str(expr2))
        return is_correct

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
                "Provide ONLY the final answer wrapped as \\boxed{{...}}.\n\n"
                "Problem: {query}\nAnswer:"
            )

        pbar = tqdm(self.dataset, desc=f"Evaluating MATH-500: acc - {correct}({0})/{total}")
        for i, ex in enumerate(pbar):
            if limit and i >= limit:
                break

            q, gold = ex["question"], ex["answer"]
            pred_text = self.generate(prompt.format(query=q), max_new_tokens=512)

            # extraction:
            pred_text_norm = self._latex_normalize(pred_text)
            pred_raw = self._extract_boxed(pred_text_norm) # list of boxes
            extract_method = "boxed"
            if pred_raw is None or not pred_raw:
                pred_raw = self._fallback_last_math_line(pred_text_norm) # list of lines or list of math like outputs
                extract_method = "last_line"
            
            # canonicalize both gold and pred for comparison
            gold_raw = ex["answer"]  # original from dataset
            pred_norm_list = list(map(self._canonicalize,pred_raw))
            gold_norm = self._canonicalize(gold_raw)
            
            # compare using sympy when available (sympy objects) else string equality
            is_correct = self.sympy_checker(pred_norm_list[-1], gold_norm) if pred_norm_list else False
            is_partially_correct = any(self.sympy_checker(pred, gold_norm) for pred in pred_norm_list)

            results.append({
                "question": q,
                "gold": gold_raw,
                "gold_norm": str(gold_norm),
                "pred_text": pred_text,
                "pred_raw": pred_raw,
                "extract_method": extract_method,
                "pred_norm": str(pred_norm_list),
                "correct": bool(is_correct),
                "partial_correct": bool(is_partially_correct),
            })

            print(results[-1])

            acc = correct / total if total > 0 else 0.0
            total += 1
            correct += int(is_correct)
            partial += int(is_partially_correct)
            pbar.set_description(f"Evaluating MATH-500: acc - {partial}({correct})/{total} ({acc:.2%})")


        pbar.close()

        return {"accuracy": acc, "total": total, "correct": correct, "details": results}