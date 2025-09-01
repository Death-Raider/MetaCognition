from datasets import load_dataset
import torch
from tqdm import tqdm
import re
from logger import logger
import os
import time
import httpx
import json
import pandas as pd

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

class GPT:
    def __init__(self, model):
        self.model_name = model
        # Load client
        print("Using OpenAI model:", self.model_name)
        print("Ensure your OPENAI_API_KEY environment variable is set.")
        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )

    def query_openai(self, messages, model=None, temperature=0.2):
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": self.model_name if model is None else model,
            "messages": messages,
            "temperature": temperature,
        }

        response = self.client.post(url, json=payload)
        data = response.json()

        # Extract headers
        headers = response.headers
        rate_info = {
            "requests_left": headers.get("x-ratelimit-remaining-requests"),
            "tokens_left": headers.get("x-ratelimit-remaining-tokens"),
            "requests_reset": headers.get("x-ratelimit-reset-requkests"),
            "tokens_reset": headers.get("x-ratelimit-reset-tokens"),
        }
        if 'choices' not in data:
            print(data)
            raise ValueError("No choices returned from OpenAI API. Check your request and model.")
        return data["choices"][0]["message"]["content"], rate_info

class GPT_Bench:
    def __init__(self, gpt_model, dataset):
        self.dataset: list[dict] = dataset
        self.gpt: GPT = gpt_model
        # Load full instruction prompt
        with open("Benchmarking/instructions.txt", "r") as f:
            self.instruction_prompt = f.read()
    
    def build_messages(self, entry):
        return [
            {"role": "system", "content": "You are a cognitive decomposition engine."},
            {"role": "user", "content": f"{self.instruction_prompt}\n\nHere is the input:\n{json.dumps(entry, indent=2)}"},
        ]

    def parse_eval_output(self,output_text):
        try:
            result = json.loads(output_text)
        except json.JSONDecodeError:
            # fallback: try to clean the text
            cleaned = output_text[output_text.find("{"):output_text.rfind("}")+1]
            try:
                result = json.loads(cleaned)
            except:
                print(f"Failed to decode JSON: {output_text}")
                return None
        return result

    def bench(self, limit=50):
        for i,entry in enumerate(self.dataset):
            print(f"Evaluating entry {i+1}/{len(self.dataset)}")
            if (limit is not None) and (i >= limit):
                break
            message = self.build_messages(entry)
            output_text, rate_information = self.gpt.query_openai(message, model="gpt-4.1", temperature=0.2)
            eval_result = self.parse_eval_output(output_text)
            if eval_result is not None:
                entry.update(eval_result)
                # logger.info(f"Eval result: {eval_result}")
            else:
                continue
            if int(rate_information.get('requests_left',0)) <=1 or int(rate_information.get('tokens_left',0)) <= 100:
                logger.warning("Rate limit reached, waiting for reset...")
                time.sleep(1)
        return self.dataset

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
                temperature=0.0,
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

    def evaluate(self, limit=None, prompt:str=None):
        total, correct = 0, 0
        results = []
        if prompt is None:
            prompt = (
                "Solve the following math problem step-by-step. "
                "Let's think step by step.\n\n"
                "Q: {query}\nA:"
            )
        pbar = tqdm(self.dataset, desc=f"Evaluating GSM8K: acc - {correct}/{total}")
        for i, ex in enumerate(pbar):
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
            # logger.info(f"Q: {q}\nG: {gold_num}\nP: {pred_text}\nCorrect: {is_correct}\n")
            
            acc = correct / total if total > 0 else 0.0
            pbar.set_description(f"Evaluating GSM8K: acc - {correct}/{total} ({acc:.2%})")        
        pbar.close()
        return {"accuracy": acc, "total": total, "correct": correct, "details": results}


def bench(model, tokenizer, prompt_instruction:str=None):
    gsm8k = GSM8K()
    bench = GSM8K_Bench(model, tokenizer, gsm8k, device="cuda" if torch.cuda.is_available() else "cpu")
    results = bench.evaluate(limit=30, prompt=prompt_instruction)  # limit for quicker test run
    print(f"GSM8K Accuracy: {results['accuracy']*100:.2f}% "
          f"({results['correct']}/{results['total']})")
    print("Running GPT on results for cognitive decomposition...")
    gpt = GPT(model="gpt-4.1")
    gpt_bench = GPT_Bench(gpt, results['details'])
    results = pd.DataFrame(gpt_bench.bench(limit=50))
    imp_columns = [
        "Logical Flow",
        "Structural Organization",
        "Consistency",
        "Factual Correctness",
        "Domain Knowledge Application",
        "Reasoning Validity",
        "Final Answer Correctness",
        "Strategy Usefulness",
        "Progress Toward Solution",
        "Partial Success Recognition",
        "Error Robustness",
        "verbosity",
        "final_comment"
    ]
    print("Cognitive decomposition results:\n", results[imp_columns].describe())

    return results
