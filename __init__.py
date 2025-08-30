# create the dataset by running the create_ds.py script
# import semi_automated_dataset_creation.create_ds as CreateDS

# train the model by running any of these trainer scripts
# import Stage_1_Basic_DPO.trainer_stage_1 as Trainer_DPO
# import Stage_5_HiPO_1Pass.trainer_stage_5 as Trainer_1Pass
# import Stage_5_HiPO_PromptGen_2Pass.trainer_stage_5 as Trainer_2Pass

# Evaluate the model by running the evaluate.py script
import Benchmarking.benchmark as bench
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
bench.bench(model=model, tokenizer=tokenizer, prompt_instruction=None)