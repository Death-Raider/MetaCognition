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
import torch
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAMES = [
    # "Stage_5_HiPO_1Pass/models_saved/Qwen",
    
    # "Stage_5_HiPO_1Pass/models_saved/model_w(1,0,0,0)",
    # "Stage_5_HiPO_1Pass/models_saved/model_w(0,1,0,0)",
    # "Stage_5_HiPO_1Pass/models_saved/model_w(0,0,1,0)",
    # "Stage_5_HiPO_1Pass/models_saved/model_w(0,0,0,1)",
    
    "Stage_5_HiPO_1Pass/models_saved/model_w(0.60,0.15,0.15,0.10)",
    "Stage_5_HiPO_1Pass/models_saved/model_w(0.20,0.50,0.20,0.10)",
    "Stage_5_HiPO_1Pass/models_saved/model_w(0.15,0.20,0.50,0.15)",
    "Stage_5_HiPO_1Pass/models_saved/model_w(0.10,0.20,0.20,0.50)",
    "Stage_5_HiPO_1Pass/models_saved/model_w(0.35,0.30,0.15,0.25)",
]

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
tokenizer.pad_token = tokenizer.eos_token
for MODEL_NAME in MODEL_NAMES:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
    
    prompt_instruction = open('Stage_5_HiPO_1Pass/instructions/instruction_few_shot.txt', 'r').read().strip()
    
    bench_results = bench.bench(model=model, tokenizer=tokenizer, prompt_instruction=prompt_instruction)
    bench_results = json.dumps(bench_results.to_dict())
    
    with open(f"{MODEL_NAME}.json","w") as f:
        f.write(bench_results)
    
    # eval_metrics["weights"].append(w[:-2].cpu().numpy().tolist())
    # eval_metrics["epoch"].append(w[-1].item())
    # eval_metrics["bench_resullts"].append(bench_results)
    # eval_metrics["loss_history"].append(loss_history_epoch)
    # eval_metrics["component_history"].append(loss_component_history)