# ========== imports ==============
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Stage_5_HiPO_1Pass.DPO import DirectPreferenceOptimization
import json
from ConfigSchema import ConfigSchema
from logger import logger

# ========== Config Loading ==============
config_schema = ConfigSchema()
with open("config.cfg", "r") as cfg:
    config = {}
    for line in cfg:
        if line.strip() and not line.startswith("#"):
            key, value = line.strip().split("=")
            config[key.strip()] = value.strip()
config_schema.from_dict(config)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Config loaded:", config_schema)
logger.info(f"Config loaded:{config_schema}")
print("Device set as:", DEVICE)
logger.info(f"Device set as:{DEVICE}")
    

with open('semi_automated_dataset_creation/processed_decomposed_dataset.jsonl', 'r') as f:
    preference = [json.loads(line) for line in f]

prompt_instruction = """
    Given a problem, generate a detailed reasoning process that includes:
        1. Redefined Query: Repharse the original query to ensure clarity.
        2. Meta-cognitive Plan: Outline a high-level strategy for solving the problem.
        3. Tactical Plan: Detail specific steps or methods to implement the meta-cognitive plan.
        4. Answer: Provide the final answer or solution to the problem.
    Question: {query}
""".strip()

# ====== Initialize DPO and DataLoader ======
dataset = preference

DPO = DirectPreferenceOptimization(config_schema.beta, DEVICE, config_schema.lr, config_schema.max_len)
DPO.set_models(config_schema.model_name)
gen_prompt_ids = DPO.tokenizer(
    prompt_instruction,
    return_tensors='pt',
    add_special_tokens=False
).input_ids.to(DPO.device)
loader = DataLoader(dataset, batch_size=config_schema.batch_size, shuffle=True, collate_fn=DPO.collate_fn)

# ====== Training loop ======
loss = torch.Tensor(0.0).to(DEVICE)

weights_Rq_only = torch.tensor([1.0, 0.0, 0.0, 0.0 ]).to(DEVICE)  # Weights for Rq span
weights_Mt_only = torch.tensor([0.0, 1.0, 0.0, 0.0]).to(DEVICE)  # Weights for Mt span
weights_Ra_only = torch.tensor([0.0, 0.0, 1.0, 0.0 ]).to(DEVICE)  # Weights for Ra span
weights_Together = torch.tensor([0.25, 0.25, 0.20, 0.30 ]).to(DEVICE)  # Weights for [Rq, Mt, Ra, R} spans

weights = weights_Together

for epoch in range(config_schema.epochs):
    total_loss = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch + 1} Loss: {loss.item():.2f}"):
        DPO.policy_optimizer.zero_grad()
        loss = DPO.dpo_loss(batch, Prompt_Instruction = gen_prompt_ids,beta = config_schema.beta, weights=weights)
        loss.backward()
        DPO.policy_optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1} Loss: {total_loss / len(loader):.4f}")
    logger.info(f"Epoch {epoch + 1} Loss: {total_loss / len(loader):.4f}")
