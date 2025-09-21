import torch
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
import Benchmarking.benchmark as bench
from Stage_5_HiPO_1Pass.trainer_stage_5 import *

# Initialize
config_schema, DEVICE, prompt_instruction, DPO, gen_prompt_ids, loader = init()
imp_columns = [
    "Logical Flow",
    "Structural Organization",
    "Consistency",
    # "Factual Correctness",
    "Domain Knowledge Application",
    "Reasoning Validity",
    # "Final Answer Correctness",
    "Strategy Usefulness",
    "Progress Toward Solution",
    "Partial Success Recognition",
    "Error Robustness",
    # "verbosity",
    # "final_comment"
]

# Define objective function
def train_with_config(config):
    # turn config into weights tensor (Rq, Mt, Ra, Y, lr, epochs)
    weights = torch.tensor([
        [config["rq"], config["mt"], config["ra"], config["y"], config["lr"], config["epochs"]]
    ]).to(DEVICE)

    eval_metrics = create_eval_metric()
    training(
        weights,
        eval_metrics,
        DPO,
        loader,
        gen_prompt_ids,
        config_schema,
        prompt_instruction,
        True,
        "individual",
    )
    results = bench.bench(model=DPO.policy_model, tokenizer=DPO.tokenizer, prompt_instruction=prompt_instruction, intrem_save_path=None, limit=30)
    results['bench'] = results['bench'].fillna("Math500")
    accuracies = results.groupby('bench')['Final Answer Correctness'].mean() / 10.0
    gsm8k = accuracies.get("GSM8K", 0.0)
    math500 = accuracies.get("Math500", 0.0)
    gpt = results[imp_columns].values().mean()

    # Report back to Ray Tune
    tune.report(gsm8k=gsm8k, math500=math500, gpt=gpt, mean_score=(gsm8k + math500) / 3.0)


# Define search space
search_space = {
    "rq": tune.uniform(0.0, 1.0),
    "mt": tune.uniform(0.0, 1.0),
    "ra": tune.uniform(0.0, 1.0),
    "y": tune.uniform(0.0, 1.0),
    "lr": tune.loguniform(1e-6, 1e-5),
    "epochs": tune.choice([1,2,3,4,5])
}

# Normalize weights inside train_with_config if needed
# (rq+mt+ra+y should sum to ~1.0)

# Setup Hyperband scheduler
scheduler = HyperBandScheduler(
    metric="mean_score",
    mode="max",
    max_t=5,              # max epochs
    reduction_factor=2
)

# Launch search
tuner = tune.Tuner(
    train_with_config,
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        num_samples=20,  # try 20 configs
    ),
    param_space=search_space,
)

results = tuner.fit()

best_result = results.get_best_result(metric="mean_score", mode="max")
print("Best config:", best_result.config)
print("Best metrics:", best_result.metrics)
