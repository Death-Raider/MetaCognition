import httpx
import os
import time
import json
from logger import logger
from tqdm import tqdm
import concurrent.futures

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
        try:
            data = response.json()
        except:
            print(f"Response issue: status={response.status_code}")
            print("Headers:", response.headers)
            print("Body:", response.text)
            data = {
                'choices':[{
                    'message':{'content':""}
                }]
            }
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
    def __init__(self, gpt_model, dataset, batch_size=10):
        self.dataset: list[dict] = dataset
        self.gpt: GPT = gpt_model
        self.batch_size = batch_size
        with open("Benchmarking/instructions.txt", "r") as f:
            self.instruction_prompt = f.read()
    
    def build_messages(self, entry):
        return [
            {"role": "system", "content": "You are a cognitive decomposition engine."},
            {"role": "user", "content": f"{self.instruction_prompt}\n\nHere is the input:\n{json.dumps(entry, indent=2)}"},
        ]

    def parse_eval_output(self, output_text):
        try:
            return json.loads(output_text)
        except json.JSONDecodeError:
            cleaned = output_text[output_text.find("{"):output_text.rfind("}")+1]
            try:
                return json.loads(cleaned)
            except:
                print(f"Failed to decode JSON: {output_text}")
                return None

    def process_entry(self, entry):
        """Helper for concurrent execution"""
        message = self.build_messages(entry)
        output_text, rate_information = self.gpt.query_openai(
            message, model="gpt-4.1", temperature=0.2
        )
        eval_result = self.parse_eval_output(output_text)
        if eval_result is not None:
            entry.update(eval_result)
        return entry, rate_information

    def bench(self, limit=50):
        results = []
        pbar = tqdm(total=min(limit, len(self.dataset)), desc="Evaluating GPT")

        for start in range(0, min(limit, len(self.dataset)), self.batch_size):
            batch = self.dataset[start:start+self.batch_size]

            # Run this batch concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
                futures = [executor.submit(self.process_entry, entry) for entry in batch]
                for f in concurrent.futures.as_completed(futures):
                    try:
                        entry, rate_info = f.result()
                        results.append(entry)
                        # crude safeguard: if nearly out of requests/tokens, pause briefly
                        if (rate_info.get("requests_left") is not None 
                            and int(rate_info["requests_left"]) <= 1):
                            time.sleep(2)
                    except Exception as e:
                        print("Error in worker:", e)
                    pbar.update(1)

        pbar.close()
        return results