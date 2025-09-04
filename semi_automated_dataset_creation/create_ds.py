from datasets import load_dataset
import os
import json
from tqdm import tqdm
import time
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import re

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

def safe_json_loads(raw_str: str):
    """
    Safer json.loads with backslash-fix.
    Tries multiple fallbacks if the first attempt fails.
    """
    try:
        # First try normal JSON parse
        return json.loads(raw_str)
    except json.JSONDecodeError:
        # Regex fix: double backslashes that aren't valid escapes
        fixed_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw_str)
        try:
            return json.loads(fixed_str)
        except json.JSONDecodeError as e:
            # Last resort: escape *all* backslashes
            safe_str = raw_str.replace("\\", "\\\\")
            return json.loads(safe_str)

def create_ds(entry, model="gpt-4.1", max_retries=1):
    for attempt in range(max_retries):
        try:
            messages = build_messages(entry)
            output_text, rate_information = query_openai(messages, model=model, temperature=0.2)
            try:
                output_json = safe_json_loads(output_text[8:-4]) # ````json and ```` removal
                output_json.update(entry)
            except json.JSONDecodeError:
                print(f"Failed to decode JSON")
                raise
            if int(rate_information.get('requests_left',0)) <=1 or int(rate_information.get('tokens_left',0)) <= 100:
                print("Rate limit reached, waiting for reset...")
                time.sleep(1)
            return output_json
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error on entry — {e}")
            time.sleep(2 ** attempt)  # exponential backoff

    return {"error": "Max retries exceeded", **entry}

def process_entries(entries, already_done, output_file, max_workers=2):
    fout_lock = Lock()   # ensure safe file writes
    pbar_lock = Lock()   # ensure tqdm is updated safely

    with open(output_file, "a") as fout, \
         ThreadPoolExecutor(max_workers=max_workers) as executor, \
         tqdm(total=len(entries)) as pbar:

        futures = []
        for entry in entries:
            if entry["query"] in already_done:
                pbar.update(1)
                continue
            futures.append(executor.submit(create_ds, entry))

        for future in as_completed(futures):
            result = future.result()
            # write result safely
            with fout_lock:
                fout.write(json.dumps(result) + "\n")
                fout.flush()
            # update tqdm safely
            with pbar_lock:
                pbar.update(1)

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
# Use either multi-threaded or single-threaded processing
# process_entries(entries, already_done, output_file, max_workers=4)

# basic looping (single-threaded)
with open(output_file, "a") as fout:
    for entry in tqdm(entries):
        if entry["query"] in already_done:
            continue
        result = create_ds(entry)
        fout.write(json.dumps(result) + "\n")
        fout.flush()  # ensure write is safe in case of crash