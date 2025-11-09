# ========== imports ==============
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Stage_5_HiPO_PromptGen_2Pass.DPO import DirectPreferenceOptimization
import json
from ConfigSchema import ConfigSchema
from logger import logger

# ========== Config Loading ==============
config_schema = ConfigSchema()
config_schema.from_file("config.cfg")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Config loaded:", config_schema)
logger.info(f"Config loaded:{config_schema}")
print("Device set as:", DEVICE)
logger.info(f"Device set as:{DEVICE}")
    
# === Example dummy dataset ===
"""
{
  "query": <original query>,
  "output_a": <model output a>,
  "M_a_text": <explanation of meta-cognitive plan>,
  "M_a_span": [M_start, M_end],
  "T_a_text": <explanation of tactical plan>,
  "T_a_span": [T_start, T_end],
  "A_a_text": <verbatim answer span>,
  "A_a_span": [A_start, A_end],
  "S_a": <short strategy label>,
  "output_b": <model output b>,
  "M_b_text": <explanation of meta-cognitive plan>,
  "M_b_span": [M_start, M_end],
  "T_b_text": <explanation of tactical plan>,
  "T_b_span": [T_start, T_end],
  "A_b_text": <verbatim answer span>,
  "A_b_span": [A_start, A_end],
  "S_b": <short strategy label>,
  "label": <original label>
}
"""
with open('semi_automated_dataset_creation/processed_decomposed_dataset.jsonl', 'r') as f:
    preference = [json.loads(line) for line in f]


# ====== Initialize DPO and DataLoader ======
dataset = preference
DPO = DirectPreferenceOptimization(config_schema.beta, DEVICE, config_schema.lr, config_schema.max_len)
DPO.set_models(config_schema.model_name)

loader = DataLoader(dataset, batch_size=config_schema.batch_size, shuffle=True, collate_fn=DPO.collate_fn)

# ====== Training loop ======
loss = torch.tensor(0.0).to(DEVICE)
for epoch in range(config_schema.epochs):
    total_loss = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch + 1} Loss: {loss.item():.2f}"):
        DPO.policy_optimizer.zero_grad()
        loss = DPO.dpo_loss(batch, beta = config_schema.beta)
        loss.backward()
        DPO.policy_optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1} Loss: {total_loss / len(loader):.4f}")
    logger.info(f"Epoch {epoch + 1} Loss: {total_loss / len(loader):.4f}")
