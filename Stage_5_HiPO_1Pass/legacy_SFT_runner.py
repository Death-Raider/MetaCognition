from Stage_5_HiPO_1Pass.HiPO_trainer import *

config_schema, DEVICE,prompt_instruction, DPO, gen_prompt_ids, loader = init()

sft_weights = torch.tensor([
    [0.00, 0.00, 0.00, 0.00, 1e-5, 5], # SFT doesnt need weights
]).to(DEVICE)

eval_metrics = create_eval_metric()
# eval_metrics = ref_model_eval(eval_metrics, DPO, prompt_instruction)
training(sft_weights, eval_metrics, DPO, loader, gen_prompt_ids, config_schema, prompt_instruction, True, 'sft')
# training_history_seq = json.dumps(eval_metrics_seq, indent=4)
# with open('Stage_5_HiPO_1Pass/training_history_sft.json', 'w') as f:
#     f.write(training_history_seq)
