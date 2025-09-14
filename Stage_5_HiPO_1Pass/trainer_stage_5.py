# ========== imports ==============
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Stage_5_HiPO_1Pass.DPO import DirectPreferenceOptimization
import json
from ConfigSchema import ConfigSchema
from logger import logger
import Benchmarking.benchmark as bench

def create_eval_metric():
    return {
    "weights": [],
    "epoch": [],
    "bench_resullts": [],
    "loss_history": [],
    "component_history": []
}

def ref_model_eval(eval_metrics, DPO, prompt_instruction=None):
    bench_results = bench.bench(model=DPO.ref_model, tokenizer=DPO.tokenizer, prompt_instruction=prompt_instruction)
    bench_results = bench_results.to_dict()

    eval_metrics["weights"].append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # no weights for reference model
    eval_metrics["epoch"].append(0)
    eval_metrics["bench_resullts"].append(bench_results)
    eval_metrics["loss_history"].append([])  # no loss history for reference model
    eval_metrics["component_history"].append([])  # no component history for reference model
    return eval_metrics

def training(
    weights: torch.Tensor, 
    eval_metrics: dict,
    DPO: DirectPreferenceOptimization, 
    loader: DataLoader, 
    gen_prompt_ids: dict, 
    config_schema: ConfigSchema, 
    prompt_instruction: str|None = None,
    reset_model: bool = True,
    method: str = 'sequential'
):
    assert method in ['sequential', 'individual', 'sft'] , "method must be either 'sequential' or 'individual'"
    if reset_model:
        DPO.set_models(config_schema.model_name)  # reset to reference model before training
    loss = torch.tensor(0.0).to(DPO.device)
    
    for w in weights:
        print(f"Training with weights: {w}")
        logger.info(f"Training with weights: {w}")
        DPO.lr = w[-2].item()
        loss_history_epoch = []
        loss_component_history = []
        scaler = torch.amp.GradScaler(DPO.device)
        for epoch in range(int(w[-1].item())):
            total_loss = 0
            total_loss_components = torch.tensor([0, 0, 0, 0], dtype=torch.bfloat16).to(DPO.device)
            data_loader = tqdm(loader, desc=f"Epoch {epoch + 1} Loss: {loss.item():.2f}")
            for batch in data_loader:
                DPO.policy_optimizer.zero_grad(set_to_none=True)
                if method == 'sft':
                    loss = DPO.sft_loss(batch, Prompt_Instruction = gen_prompt_ids)
                    loss_components = (0,0,0,0)
                else:
                    loss, loss_components = DPO.dpo_loss(batch, Prompt_Instruction = gen_prompt_ids,beta = config_schema.beta, weights=w[:-2])
                loss_M, loss_T, loss_A, loss_MTAS = loss_components
                total_loss_components += torch.stack([
                    loss_M.detach().mean(),
                    loss_T.detach().mean(),
                    loss_A.detach().mean(),
                    loss_MTAS.detach().mean()
                ]).to(DPO.device)
                loss.backward()
                DPO.policy_optimizer.step()
                total_loss += loss.item()
                data_loader.set_description(f"Epoch {epoch + 1} Loss: {loss.item():.4f}")
            loss_history_epoch.append(total_loss / len(loader))
            loss_component_history.append(
                (total_loss_components / len(loader)).to(torch.float32).cpu().numpy().tolist()
            )
            print(f"Epoch {epoch + 1} Loss: {total_loss / len(loader):.4f}")
            logger.info(f"Epoch {epoch + 1} Loss: {total_loss / len(loader):.4f}")
            # checkpoint after each epoch
            DPO.policy_model.save_pretrained(f"Stage_5_HiPO_1Pass/models_saved/model_w{w}", from_pt=True)
        # benchmark after training with each weight configuration
        bench_results = bench.bench(model=DPO.policy_model, tokenizer=DPO.tokenizer, prompt_instruction=prompt_instruction)
        bench_results = bench_results.to_dict()

        if method == 'individual':
            DPO.set_models(config_schema.model_name)  # reset to reference model before next weight config

        eval_metrics["weights"].append(w[:-2].cpu().numpy().tolist())
        eval_metrics["epoch"].append(w[-1].item())
        eval_metrics["bench_resullts"].append(bench_results)
        eval_metrics["loss_history"].append(loss_history_epoch)
        eval_metrics["component_history"].append(loss_component_history)
    return eval_metrics

def init():
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

    prompt_instruction = open('Stage_5_HiPO_1Pass/instructions/instruction_few_shot.txt', 'r').read().strip()

    # ====== Initialize DPO and DataLoader ======
    limit = 10000
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
    return config_schema, DEVICE,prompt_instruction, DPO, gen_prompt_ids, loader
