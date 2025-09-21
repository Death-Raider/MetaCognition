# create the dataset by running the create_ds.py script
# import semi_automated_dataset_creation.create_ds as CreateDS

# train the model by running any of these trainer scripts
# import Stage_1_Basic_DPO.trainer_stage_1 as Trainer_DPO
# import Stage_5_HiPO_1Pass.trainer_stage_5 as Trainer_1Pass

# import Stage_5_HiPO_1Pass.SFT_runner as sft
# import Stage_5_HiPO_1Pass.ablation_runner as ablation
# import Stage_5_HiPO_1Pass.sequential_runner as sequential_run

# import Stage_5_HiPO_PromptGen_2Pass.trainer_stage_5 as Trainer_2Pass

# Evaluate the model by running the evaluate.py script
import Benchmarking.benchmark as bench
from transformers import AutoModelForCausalLM, AutoTokenizer
# from huggingface_hub import snapshot_download, scan_cache_dir
import torch
import json
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
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
    ]
}

for model_type, model_list in MODELS.items():
    # pick tokenizer
    if model_type == 'Qwen':
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
    elif model_type == 'Mistral':
        tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')
    tokenizer.pad_token = tokenizer.eos_token

    for model_name in model_list:
        print(f"Loading {model_name} ...")

        # load model
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

        # run benchmark
        prompt_instruction = open('Stage_5_HiPO_1Pass/instructions/instruction_few_shot.txt', 'r').read().strip()
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

    
    # eval_metrics["weights"].append(w[:-2].cpu().numpy().tolist())
    # eval_metrics["epoch"].append(w[-1].item())
    # eval_metrics["bench_resullts"].append(bench_results)
    # eval_metrics["loss_history"].append(loss_history_epoch)
    # eval_metrics["component_history"].append(loss_component_history)
