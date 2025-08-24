# ========== imports ==============
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
# from torchviz import make_dot
from Stage_1_Basic_DPO.PreferenceDataLoader import PreferenceDataLoader
from Stage_1_Basic_DPO.DPO import DirectPreferenceOptimization
import json
from ConfigSchema import ConfigSchema
from datasets import load_dataset
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

# === Fixed prompts ===
STRATS = ["Self Verify", "Chain of Thought"]
FIXED_STRATEGY = STRATS[1]
print("Using Stratergy:", FIXED_STRATEGY)
logger.info(f"Using Strategy:{FIXED_STRATEGY}")

with open('semi_automated_dataset_creation/processed_decomposed_dataset.jsonl', 'r') as f:
    preference = [json.loads(line) for line in f]

# ====== Initialize DPO and DataLoader ======
dataset = PreferenceDataLoader(preference, strat=FIXED_STRATEGY) # augment the dummy dataset with the fixed strategy
DPO = DirectPreferenceOptimization(config_schema.beta, DEVICE , config_schema.lr, config_schema.max_len)
DPO.set_models(config_schema.model_name)
loader = DataLoader(dataset, batch_size=config_schema.batch_size, shuffle=True, collate_fn=DPO.collate_fn)
DPO.test_model_capability(dataset, FIXED_STRATEGY)

# ====== Training loop ======
for epoch in range(config_schema.epochs):
    total_loss = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch + 1}"):
        DPO.policy_optimizer.zero_grad()
        loss = DPO.dpo_loss(batch['prompt_inputs'], batch['chosen_outputs'], batch['rejected_outputs'])
        loss.backward()
        DPO.policy_optimizer.step()
        total_loss += loss.item()
    DPO.test_model_capability(dataset, FIXED_STRATEGY)
    print(f"Epoch {epoch + 1} Loss: {total_loss / len(loader):.4f}")
    logger.info(f"Epoch {epoch + 1} Loss: {total_loss / len(loader):.4f}")
