import httpx
import os
import time
import json
from logger import logger

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