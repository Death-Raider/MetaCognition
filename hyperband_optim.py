import torch
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
import Benchmarking.benchmark as bench
from Stage_5_HiPO_1Pass.trainer_stage_5 import *

# Initialize
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
    config_schema, DEVICE, prompt_instruction, DPO, gen_prompt_ids, loader = init()
    # turn config into weights tensor (Rq, Mt, Ra, Y, lr, epochs)
    sum_conf  = config["rq"] + config["mt"] + config["ra"] + config["y"]
    
    weights = torch.tensor([
        [config["rq"]/sum_conf, config["mt"]/sum_conf, config["ra"]/sum_conf, config["y"]/sum_conf, config["lr"], config["epochs"]]
    ]).to(DEVICE)

    eval_metrics = create_eval_metric()

    def trial_bench(model, tokenizer, total_loss, component_loss):
        results = bench.bench(model=model, tokenizer=tokenizer, prompt_instruction=prompt_instruction, intrem_save_path=None, limit=30)
        results['bench'] = results['bench'].fillna("Math500")
        accuracies = results.groupby('bench')['Final Answer Correctness'].mean() / 10.0
        gsm8k = accuracies.get("GSM8K", 0.0)
        math500 = accuracies.get("Math500", 0.0)
        gpt = results[imp_columns].values().mean()

        # Report back to Ray Tune
        tune.report(gsm8k=gsm8k, math500=math500, gpt=gpt, mean_score=(gsm8k + math500) / 2.0)

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
        save=False,
        callbacks={
            'epoch': trial_bench
        }
    )


# Define search space
search_space = {
    "rq": tune.uniform(0.0, 1.0),
    "mt": tune.uniform(0.0, 1.0),
    "ra": tune.uniform(0.0, 1.0),
    "y": tune.uniform(0.0, 1.0),
    "lr": tune.loguniform(1e-6, 1e-5),
    "epochs": tune.choice([1,2,3,4,5])
}

# Setup Hyperband scheduler
scheduler = HyperBandScheduler(
    metric="mean_score",
    mode="max",
    max_t=5,
    reduction_factor=3
)

trainable_with_resources = tune.with_resources(
    train_with_config,
    resources={"cpu": 0, "gpu": 1} # Adjust as needed
)
# Launch search
tuner = tune.Tuner(
    trainable_with_resources,
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
