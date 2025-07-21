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

def safe_judge(entry, max_retries=5):
    for attempt in range(max_retries):
        try:
            messages = build_messages(entry)
            output_text, rate_information = query_openai(messages, model="gpt-4.1", temperature=0.2)
            try:
                output_json = json.loads(output_text[8:-4])
            except json.JSONDecodeError:
                print(f"Failed to decode JSON: {output_text}")
                raise
            if int(rate_information.get('requests_left',0)) <=1 or int(rate_information.get('tokens_left',0)) <= 100:
                print("Rate limit reached, waiting for reset...")
                time.sleep(1)
            return {
                "query": entry["query"],
                "output_a": entry["output_a"],
                "M_a_text": output_json.get("M_a_text", ""),
                "T_a_text": output_json.get("T_a_text", ""),
                "A_a_text": output_json.get("A_a_text", ""),
                "M_a_span": output_json.get("M_a_span", []),
                "T_a_span": output_json.get("T_a_span", []),
                "A_a_span": output_json.get("A_a_span", []),
                "S_a": output_json.get("S_a", ""),
                "output_b": entry["output_b"],
                "M_b_text": output_json.get("M_b_text", ""),
                "T_b_text": output_json.get("T_b_text", ""),
                "A_b_text": output_json.get("A_b_text", ""),
                "M_b_span": output_json.get("M_b_span", []),
                "T_b_span": output_json.get("T_b_span", []),
                "A_b_span": output_json.get("A_b_span", []),
                "S_b": output_json.get("S_b", ""),
                "label": entry["label"],
            }

        except Exception as e:
            print(f"Attempt {attempt + 1}: Error on entry — {e}")
            time.sleep(2 ** attempt)  # exponential backoff

    return {"error": "Max retries exceeded", **entry}

# Load dataset
raw_data = load_dataset("nvidia/HelpSteer2", data_dir="preference")['train']
entries = [
    {
        "query": x["prompt"],
        "output_a": x["response_1"],
        "output_b": x["response_2"],
        "label": x["preference_strength"],
    }
    for x in raw_data
]

# Output file (resumable)
output_file = "processed_decomposed_dataset.jsonl"
already_done = set()
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        already_done = {json.loads(line[:-2])["query"] for line in f}

print(f"Already processed {len(already_done)} entries.")

# Run loop
with open(output_file, "a") as fout:
    for entry in tqdm(entries):
        if entry["query"] in already_done:
            continue
        result = safe_judge(entry)
        fout.write(json.dumps(result) + ",\n")
        fout.flush()  # ensure write is safe in case of crash