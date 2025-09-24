from Benchmarking.GSM8K_bench import GSM8K, GSM8K_Bench
from Benchmarking.GPT_bench import GPT, GPT_Bench
from Benchmarking.Math500_bench import MATH500, MATH500_Bench
from Benchmarking.additional_bench import AIME24, Gaokao2023En, MinervaMath 

import torch
from logger import logger
import pandas as pd
import json

# HuggingFaceH4/MATH-500
# gsm8k
# math-ai/aime24
# MARIO-Math-Reasoning/Gaokao2023-Math-En
# math-ai/minervamath

def bench(model, tokenizer, prompt_instruction:str=None, intrem_save_path=None, limits=None, batch_size=None):
    if (not limits) or (limits == 0):
        limits = 500
    if (not batch_size) or (batch_size == 0):
        batch_size = 20
    
    gsm8k = GSM8K()
    bench_gsm = GSM8K_Bench(model, tokenizer, gsm8k, device="cuda" if torch.cuda.is_available() else "cpu")
    results_gsm = bench_gsm.evaluate(limit=limits, prompt=prompt_instruction, batch_size=batch_size)  # limit for quicker test run
    
    math_500 = MATH500()
    bench_math = MATH500_Bench(model, tokenizer, math_500, device="cuda" if torch.cuda.is_available() else "cpu")
    results_math = bench_math.evaluate(limit=limits, prompt=prompt_instruction, batch_size=batch_size)  # limit for quicker test run

    aime = AIME24()
    bench_aime = MATH500_Bench(model, tokenizer, aime, 'cuda')
    results_aime = bench_aime.evaluate(limit=limits, prompt=prompt_instruction, batch_size=batch_size)

    gk = Gaokao2023En('train')
    bench_gk = MATH500_Bench(model, tokenizer, gk, 'cuda')
    results_gk = bench_gk.evaluate(limit=limits, prompt=prompt_instruction, batch_size=batch_size)

    mm = MinervaMath() 
    bench_mm = MATH500_Bench(model, tokenizer, mm, 'cuda')
    results_mm = bench_mm.evaluate(limit=limits, prompt=prompt_instruction, batch_size=batch_size)
    
    
    results = results_gsm['details'] + results_math['details'] + results_aime['details'] + results_gk['details'] + results_mm['details']
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
