# ========== imports ==============
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Stage_5_HiPO_1Pass.DPO import DirectPreferenceOptimization
import json
from ConfigSchema import ConfigSchema
from logger import logger
import Benchmarking.benchmark as bench

# ========== Config Loading ==============
config_schema = ConfigSchema()
config_schema.from_file("config.cfg")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Config loaded:", config_schema)
logger.info(f"Config loaded:{config_schema}")
print("Device set as:", DEVICE)
logger.info(f"Device set as:{DEVICE}")
    

with open('semi_automated_dataset_creation/processed_decomposed_dataset.jsonl', 'r') as f:
    preference = [json.loads(line) for line in f]

prompt_instruction = open('Stage_5_HiPO_1Pass/instructions/instruction_cot.txt', 'r').read().strip()

# ====== Initialize DPO and DataLoader ======
limit = 100
dataset = preference[:limit]
for entry in dataset:
    entry["new_output_a"] = entry['Ra_a'] + "\n" + entry['Mt_a'] + "\n" + entry["Rq_a"]
    entry["new_output_b"] = entry['Ra_b'] + "\n" + entry['Mt_b'] + "\n" + entry["Rq_b"]

DPO = DirectPreferenceOptimization(config_schema.beta, DEVICE, config_schema.lr, config_schema.max_len)
DPO.set_models(config_schema.model_name)
gen_prompt_ids = DPO.tokenizer(
    prompt_instruction,
    return_tensors='pt',
    add_special_tokens=False
)
gen_prompt_ids = {k: v.to(DEVICE) for k, v in gen_prompt_ids.items()}
loader = DataLoader(dataset, batch_size=config_schema.batch_size, shuffle=True, collate_fn=DPO.collate_fn)

# ====== Training loop ======
loss = torch.tensor(0.0).to(DEVICE)

# ====== Individual weight configurations ======
# for training a model for specific weight configuration
# weights_Rq_only = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1e-5, 5]]).to(DEVICE)  # Weights for Rq span
# weights_Mt_only = torch.tensor([[0.0, 1.0, 0.0, 0.0, 1e-5, 5]]).to(DEVICE)  # Weights for Mt span
# weights_Ra_only = torch.tensor([[0.0, 0.0, 1.0, 0.0, 1e-5, 5]]).to(DEVICE)  # Weights for Ra span
# weights_Together = torch.tensor([[0.25, 0.25, 0.20, 0.30, 1e-5, 5]]).to(DEVICE)  # Weights for [Rq, Mt, Ra, R, lr, epochs} spans
# weights = weights_Rq_only

# ======= Joint weight configurations ======
# for training a model on all weight configurations one after another
# w ∈ R^6 where w = [w_Rq, w_Mt, w_Ra, w_R, lr, epochs]
weights = torch.tensor([
    [0.50, 0.20, 0.20, 0.20, 1e-5, 5],
    [0.20, 0.20, 0.20, 0.50, 1e-5, 5],
    [0.30, 0.20, 0.30, 0.30, 5e-6, 5],
    [0.25, 0.25, 0.20, 0.30, 5e-6, 5]
]).to(DEVICE)

eval_metrics = {
    "weights": [],
    "epoch": [],
    "bench_resullts": [],
    "loss_history": []
}

#benchmark reference model before training
print("Benchmarking reference model before training...")
bench_results = bench.bench(model=DPO.ref_model, tokenizer=DPO.tokenizer, prompt_instruction=prompt_instruction)
bench_results = bench_results.to_dict()

eval_metrics["weights"].append([0.0, 0.0, 0.0, 0.0])  # no weights for reference model
eval_metrics["epoch"].append(0)
eval_metrics["bench_resullts"].append(bench_results)
eval_metrics["loss_history"].append([])  # no loss history for reference model

print("")
print("Starting training...")

training_history:list[dict] = []

for w in weights:
    print(f"Training with weights: {w}")
    logger.info(f"Training with weights: {w}")
    DPO.lr = w[-2]
    loss_history_epoch = []
    for epoch in range(w[-1]):
        total_loss = 0
        data_loader = tqdm(loader, desc=f"Epoch {epoch + 1} Loss: {loss.item():.2f}")
        for batch in data_loader:
            DPO.policy_optimizer.zero_grad()
            loss = DPO.dpo_loss(batch, Prompt_Instruction = gen_prompt_ids,beta = config_schema.beta, weights=w[:-2])
            loss.backward()
            DPO.policy_optimizer.step()
            total_loss += loss.item()
            data_loader.set_description(f"Epoch {epoch + 1} Loss: {loss.item():.4f}")
        loss_history_epoch.append(total_loss / len(loader))
        print(f"Epoch {epoch + 1} Loss: {total_loss / len(loader):.4f}")
        logger.info(f"Epoch {epoch + 1} Loss: {total_loss / len(loader):.4f}")
    # benchmark after training with each weight configuration
    DPO.policy_model.save_pretrained(f"Stage_5_HiPO_1Pass/model_w{w}", from_pt=True) 
    bench_results = bench.bench(model=DPO.policy_model, tokenizer=DPO.tokenizer, prompt_instruction=prompt_instruction)
    bench_results = bench_results.to_dict()

    eval_metrics["weights"].append(w[:-2].cpu().numpy().tolist())
    eval_metrics["epoch"].append(w[-1].item())
    eval_metrics["bench_resullts"].append(bench_results)
    eval_metrics["loss_history"].append(loss_history_epoch)

training_history.append(eval_metrics)

with open('Stage_5_HiPO_1Pass/training_history.json', 'w') as f:
    json.dump(training_history, f, indent=4)