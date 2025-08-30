from datasets import load_dataset
import os
import json
from tqdm import tqdm
import time
import httpx

# Load client
client = httpx.Client(
    headers={
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json"
    },
    timeout=30.0
)

# Load full instruction prompt
with open("instructions.txt", "r") as f:
    instruction_prompt = f.read()

def build_messages(entry):
    return [
        {"role": "system", "content": "You are a cognitive decomposition engine."},
        {"role": "user", "content": f"{instruction_prompt}\n\nHere is the input:\n{json.dumps(entry, indent=2)}"},
    ]

def query_openai(messages, model="gpt-4.1", temperature=0.2):
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    response = client.post(url, json=payload)
    data = response.json()

    # Extract headers
    headers = response.headers
    rate_info = {
        "requests_left": headers.get("x-ratelimit-remaining-requests"),
        "tokens_left": headers.get("x-ratelimit-remaining-tokens"),
        "requests_reset": headers.get("x-ratelimit-reset-requests"),
        "tokens_reset": headers.get("x-ratelimit-reset-tokens"),
    }
    if 'choices' not in data:
        print(data)
        raise ValueError("No choices returned from OpenAI API. Check your request and model.")
    return data["choices"][0]["message"]["content"], rate_info

def create_ds(entry, model="gpt-4.1", max_retries=5):
    for attempt in range(max_retries):
        try:
            messages = build_messages(entry)
            output_text, rate_information = query_openai(messages, model=model, temperature=0.2)
            try:
                output_json = json.loads(output_text[8:-4]) # ````json and ```` removal
                output_json.update(entry)
            except json.JSONDecodeError:
                print(f"Failed to decode JSON: {output_text}")
                raise
            if int(rate_information.get('requests_left',0)) <=1 or int(rate_information.get('tokens_left',0)) <= 100:
                print("Rate limit reached, waiting for reset...")
                time.sleep(1)
            return output_json
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error on entry — {e}")
            time.sleep(2 ** attempt)  # exponential backoff

    return {"error": "Max retries exceeded", **entry}

# Load dataset
raw_data = load_dataset("prhegde/preference-data-math-stack-exchange")['train']
limit = 1000
entries = [
    {
        "query": x["question"],
        "output_a": x["chosen"],
        "output_b": x["rejected"],
    }
    for x in raw_data
]
entries = entries[:limit]

# Output file (resumable)
output_file = "processed_decomposed_dataset.jsonl"
already_done = set()
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        already_done = {json.loads(line)["query"] for line in f}

print(f"Already processed {len(already_done)} entries.")

# Run loop
with open(output_file, "a") as fout:
    for entry in tqdm(entries):
        if entry["query"] in already_done:
            continue
        result = create_ds(entry)
        fout.write(json.dumps(result) + "\n")
        fout.flush()  # ensure write is safe in case of crash