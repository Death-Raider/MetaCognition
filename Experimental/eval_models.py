# Evaluate the models
import Benchmarking.benchmark as bench
from transformers import AutoModelForCausalLM, AutoTokenizer
# from huggingface_hub import snapshot_download, scan_cache_dir
import torch
import json
import os

PATH_PROMPT_INSTRUCTION = 'Stage_5_HiPO_1Pass/instructions/instruction_few_shot.txt'

device = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    'Qwen': [
        "Death-Raider/HiPO-Qwen2.5-Ra-only",
        "Death-Raider/HiPO-Qwen2.5-Mt-only",
        "Death-Raider/HiPO-Qwen2.5-Rq-only",
        "Death-Raider/HiPO-Qwen2.5-seq-0",
        "Death-Raider/HiPO-Qwen2.5-seq-1",
        "Death-Raider/HiPO-Qwen2.5-seq-2",
        "Death-Raider/HiPO-Qwen2.5-seq-3",
        "Death-Raider/HiPO-Qwen2.5-seq-4",
        'Death-Raider/DPO-Qwen',
        'Death-Raider/SFT-Qwen',
    ],
    'Mistral': [
        "Death-Raider/HiPO-Mistral-seq-0",
        "Death-Raider/HiPO-Mistral-seq-1",
        "Death-Raider/HiPO-Mistral-seq-2",
        "Death-Raider/HiPO-Mistral-seq-3",
        "Death-Raider/HiPO-Mistral-seq-4",
        "Death-Raider/HiPO-Mistral-Ra-only",
        "Death-Raider/HiPO-Mistral-Mt-only",
        "Death-Raider/HiPO-Mistral-Rq-only",
        "Death-Raider/SFT-Mistral",
        "Death-Raider/DPO-Mistral",
    ],
    'llama': [
        "Death-Raider/HiPO-Llama3.1-seq-0",
        "Death-Raider/HiPO-Llama3.1-seq-1",
        "Death-Raider/HiPO-Llama3.1-seq-2",
        "Death-Raider/HiPO-Llama3.1-eq-ind",
        "Death-Raider/HiPO-Llama3.1-Mt-bias-ind",
        "Death-Raider/HiPO-Llama3.1-Rq-bias-ind",
        "Death-Raider/HiPO-Llama3.1-Ra-only",
        "Death-Raider/HiPO-Llama3.1-Mt-only",
        "Death-Raider/HiPO-Llama3.1-Rq-only",
        "Death-Raider/DPO-Llama3.1",
    ]
}

for model_type, model_list in MODELS.items():
    # pick tokenizer
    if model_type == 'Qwen':
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
    elif model_type == 'Mistral':
        tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')
    elif model_type == 'llama':
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
    tokenizer.pad_token = tokenizer.eos_token

    for model_name in model_list:
        print(f"Loading {model_name} ...")
    
        # load model
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

        # run benchmark
        if 'SFT' in model_name or 'DPO' in model_name:
            prompt_instruction = None
        else:
            prompt_instruction = open(PATH_PROMPT_INSTRUCTION, 'r').read().strip()
        bench_results = bench.bench(model=model, tokenizer=tokenizer, prompt_instruction=prompt_instruction)
        bench_results = json.dumps(bench_results.to_dict())

        # save results
        with open(f"final_results/{model_type}_{model_name.replace('/','_')}.json","w") as f:
            f.write(bench_results)

        # cleanup
        del model
        torch.cuda.empty_cache()
        os.system(f"rm -rf {os.environ['HF_HOME']}/*")
        print(f"Deleted {model_name} from cache\n")
