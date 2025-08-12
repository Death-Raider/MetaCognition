import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM

def gen_prompt_from_query(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    max_new_tokens=64,
    gen_prompt_ids=None,
    instruction="Generate a concise reasoning prompt to answer the following question:"
):
    with torch.no_grad():
    # Tokenize instruction if not provided
        if gen_prompt_ids is None:
            gen_prompt_ids = tokenizer(
                instruction,
                return_tensors='pt',
                add_special_tokens=False
            ).input_ids.to(input_ids.device)
        # Combine instruction and query
        full_input_ids = torch.cat(
            [gen_prompt_ids.repeat(input_ids.size(0), 1), input_ids], dim=1
        )
        full_attention_mask = torch.cat(
            [
                torch.ones(
                    (input_ids.size(0), gen_prompt_ids.size(1)),
                    dtype=attention_mask.dtype,
                    device=input_ids.device,
                ),
                attention_mask,
            ],
            dim=1,
        )

        # Generate continuation (the self-prompt)
        outputs = model.generate(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=5,
            temperature=0.7,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=model.config.eos_token_id,
        )
        # Query may be deeply embedded so we dont need to split out the query.
        prompt_ids = outputs.sequences[:, full_input_ids.size(1):]  # Get only the generated part

        return {
            "input_ids": prompt_ids,
            "attention_mask": torch.ones_like(prompt_ids),
            # "log_probs": compute_generation_logprobs(outputs, full_input_ids.size(1)),
            "full_input_ids": full_input_ids,
        }

def batch_prompts(prompts, pad_token_id):
    # prompts is a list of dicts returned by gen_prompt_from_query
    input_ids_list = [p['input_ids'].squeeze(0) for p in prompts]
    attn_mask_list = [p['attention_mask'].squeeze(0) for p in prompts]

    # pad_sequence will pad to the longest sequence in the batch
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
    attn_mask = pad_sequence(attn_mask_list, batch_first=True, padding_value=0)

    return {"input_ids": input_ids, "attention_mask": attn_mask}

model_name = 'Qwen/Qwen2-7B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

strategies = [
    'Supportive Stepwise Guidance',
    'Define, List, and Exemplify',
    'Stepwise Procedural Guide',
    'Enumerate Assistance Methods',
    'Intro Then List',
    'Sequential Explanation with Contrast',
    'Intro + Thematic List with Explanations',
    'Structured Reflective Statement',
    "List, Reasoning, Answer"
]
prompt_list = [
    "Generate a prompt using '{strategy}' without any extra output. Only write the prompt encoding the question: ",
    "Generate a prompt using the '{strategy}' format to guide a low-capability agent in solving a problem step-by-step without revealing the answer. You must ONLY output the prompt. If you include anything other than the prompt, or if the prompt contains the answer, your response will be considered invalid and rejected.",
    "Write a prompt using '{strategy}' that instructs a weaker agent to solve the problem step-by-step without giving the answer. Output only the prompt text—no explanations, no extra content.",
    "Using '{strategy}', create a prompt that helps a low-skill agent work through the problem in steps. Do not provide the solution or anything except the prompt itself.",
    "Construct a prompt in the '{strategy}' style that outlines a step-by-step method for a less-capable agent to follow. The prompt must not contain the answer and must be the only output.",
    "Produce a '{strategy}' prompt designed to guide a minimal-capability agent through solving the problem step-by-step. Absolutely no answers or extra commentary—output the prompt alone.",
    "Generate only the prompt in '{strategy}' style to teach a low-level agent how to approach the problem in steps. If any answer or additional text appears, the output is invalid."

]

pairs = [
    (strategy, prompt) for prompt in prompt_list for strategy in strategies
]
query = "why is 49 a prime number?"
query = tokenizer(
    query,
    return_tensors='pt',
    add_special_tokens=False
).to(model.device)

out = [
    gen_prompt_from_query(
                            model, 
                            tokenizer, 
                            query['input_ids'], 
                            query['attention_mask'], 
                            max_new_tokens=256, 
                            instruction=prompt_inst.format(strategy=strat)
                        ) 
    for strat,prompt_inst in pairs
]
# [strat_index, {input_ids[0]}] -> {input_ids[strat_index]}
new_out = batch_prompts([out],tokenizer.pad_token_id)
prompt = [tokenizer.decode(new_out['input_ids'][i], skip_special_tokens=True) for i in range(len(pairs))]
with open('PromptGenerator/prompts.txt', 'w') as f:
    for (strat, inp_prompt), prompt_txt in zip(pairs, prompt):
        f.write(f"Strategy: '{strat}'\nInput Prompt: '{inp_prompt}'\nPrompt: '{prompt_txt}'\n{'-'*40}\n")

