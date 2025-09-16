from Benchmarking.GSM8K_bench import GSM8K, GSM8K_Bench
from Benchmarking.GPT_bench import GPT, GPT_Bench
from Benchmarking.Math500_bench import MATH500, MATH500_Bench

import torch
from logger import logger
import pandas as pd
import json
def bench(model, tokenizer, prompt_instruction:str=None, intrem_save_path=None):
    limits = 30
    gsm8k = GSM8K()
    bench_gsm = GSM8K_Bench(model, tokenizer, gsm8k, device="cuda" if torch.cuda.is_available() else "cpu")
    results_gsm = bench_gsm.evaluate(limit=limits, prompt=prompt_instruction)  # limit for quicker test run
    print(f"GSM8K Accuracy: {results_gsm['accuracy']*100:.2f}% "
          f"({results_gsm['correct']}/{results_gsm['total']})")
    
    math_500 = MATH500()
    bench_math = MATH500_Bench(model, tokenizer, math_500, device="cuda" if torch.cuda.is_available() else "cpu")
    results_math = bench_math.evaluate(limit=limits, prompt=prompt_instruction)  # limit for quicker test run
    print(f"MATH500 Accuracy: {results_math['accuracy']*100:.2f}% "
          f"({results_math['correct']}/{results_math['total']})")
    
    results = results_math['details'] + results_gsm['details']
    if intrem_save_path is not None:
        with open(f'{intrem_save_path}/base_results.jsonl', "a") as f:
            f.write(json.dumps(results))
            
    print("Running GPT on results for cognitive decomposition...")
    gpt = GPT(model="gpt-4.1")
    gpt_bench = GPT_Bench(gpt, results)
    results = pd.DataFrame(gpt_bench.bench(limit=len(results)))
    imp_columns = [
        "Logical Flow",
        "Structural Organization",
        "Consistency",
        "Factual Correctness",
        "Domain Knowledge Application",
        "Reasoning Validity",
        "Final Answer Correctness",
        "Strategy Usefulness",
        "Progress Toward Solution",
        "Partial Success Recognition",
        "Error Robustness",
        "verbosity",
        "final_comment"
    ]
    if intrem_save_path is not None:
        with open(f'{intrem_save_path}/GPT_results.jsonl', "a") as f:
            f.write(json.dumps(results.to_dict(), indent=4))
    print("Cognitive decomposition results:\n", results[imp_columns].describe())
    print("overall accuracy: ", results["Final Answer Correctness"].mean())
    return results
