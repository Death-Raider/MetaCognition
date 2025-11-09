from Stage_5_HiPO_1Pass.HiPO_trainer import *

config_schema, DEVICE,prompt_instruction, DPO, gen_prompt_ids, loader = init()

print("Starting Sequential Study...")
# w ∈ R^6 where w = [w_Rq, w_Mt, w_Ra, w_R, lr, epochs]
weights = torch.tensor([
    [0.60, 0.15, 0.15, 0.10, 1e-5, 5], # Rq-bias -> stronger query alignment
    [0.20, 0.50, 0.20, 0.10, 8e-6, 5], # Mt-bias -> more reasoning
    [0.15, 0.20, 0.50, 0.15, 5e-6, 5], # Ra-bias -> force correct final answers
    [0.10, 0.20, 0.20, 0.50, 1e-6, 5], # Y/full-bias -> overall coherence
    [0.35, 0.30, 0.15, 0.25, 5e-6, 5], # Balanced with slight Rq+Mt bias
]).to(DEVICE)

eval_metrics = create_eval_metric()
training(weights, eval_metrics, DPO, loader, gen_prompt_ids, config_schema, prompt_instruction,True, 'sequential')