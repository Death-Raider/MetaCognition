from Stage_5_HiPO_1Pass.trainer_stage_5 import *


config_schema, DEVICE,prompt_instruction, DPO, gen_prompt_ids, loader = init()

print("Starting Ablation Study...")
weights = torch.tensor([
    [0.00, 0.00, 0.00, 1.00, 1e-5, 5], # standard DPO
    [1.00, 0.00, 0.00, 0.00, 1e-5, 5], # Rq Only
    [0.00, 1.00, 0.00, 0.00, 1e-5, 5], # Mt Only
    [0.00, 0.00, 1.00, 0.00, 1e-5, 5], # Ra Only
    [0.60, 0.15, 0.15, 0.10, 1e-5, 5], # Rq-bias -> stronger query alignment
    [0.20, 0.50, 0.20, 0.10, 8e-6, 5], # Mt-bias -> more reasoning
    [0.15, 0.20, 0.50, 0.15, 5e-6, 5], # Ra-bias -> force correct final answers
    [0.10, 0.20, 0.20, 0.50, 1e-5, 5], # Y/full-bias -> overall coherence
]).to(DEVICE)

eval_metrics = create_eval_metric()
eval_metrics = ref_model_eval(eval_metrics, DPO, prompt_instruction, intrem_save_path='Stage_5_HiPO_1Pass')
eval_metrics_ind = training(weights, eval_metrics, DPO, loader, gen_prompt_ids, config_schema, prompt_instruction, True, 'individual')
training_history_ind = json.dumps(eval_metrics, indent=4)
with open('Stage_5_HiPO_1Pass/training_history_ind.json', 'w') as f:
    f.write(training_history_ind)