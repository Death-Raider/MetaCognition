import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
import json 

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
        ).to(DEVICE)
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
        ).to(DEVICE)

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
            "input_ids": prompt_ids.to(DEVICE),
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'Qwen/Qwen2-7B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(DEVICE)

print(f"Using model: {model_name} on device: {DEVICE}")

strategies = [
    'Supportive Stepwise Guidance',
    'Stepwise Procedural Guide',
    'Define, List, and Exemplify',
    'Enumerate Assistance Methods',
]
prompt_list = [
    "Generate a prompt using the '{strategy}' format to guide a low-capability agent in solving a problem step-by-step without revealing the answer. You must ONLY output the prompt. If you include anything other than the prompt, or if the prompt contains the answer, your response will be considered invalid and rejected.",
    "Using '{strategy}', create a prompt that helps a low-skill agent work through the problem in steps. Do not provide the solution or anything except the prompt itself.",
]

query_set = {
    "math_reasoning": [
        "Determine whether 121 is a prime number.",
        "Find the area of a triangle with base 8 cm and height 5 cm.",
        "If a train travels 60 km in 1.5 hours, what is its average speed?"
    ],
    "general_knowledge": [
        "Who wrote the play 'Romeo and Juliet'?",
        "What is the capital city of Canada?",
        "Name three countries that share a border with Germany."
    ],
    "commonsense_reasoning": [
        "If you leave ice cubes out in the sun, what will happen after 10 minutes?",
        "Why should you not touch an electrical socket with wet hands?",
        "If it is raining outside, what might people carry with them?"
    ],
    "procedural_tasks": [
        "Explain how to change a flat bicycle tire.",
        "Give step-by-step instructions to bake a chocolate cake.",
        "Describe the process of renewing a passport."
    ],
    "creative_generation": [
        "Write the opening sentence of a mystery novel set in a small fishing village.",
        "Invent a new sport and explain how it is played.",
        "Create a tagline for a company selling eco-friendly shoes."
    ],
    "classification_identification": [
        "Decide whether this email is spam: 'Congratulations! You’ve won a $1000 gift card. Click here to claim.'",
        "Classify the tone of this sentence: 'I can't believe you did that!'",
        "Identify whether the following sentence is fact or opinion: 'Chocolate ice cream is the best flavor.'"
    ],
    "multi_step_logic": [
        "You have 3 red balls and 2 blue balls in a bag. If you take two balls without looking, what is the probability they are both red?",
        "Solve this riddle: 'I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?'",
        "If Alice is older than Bob, and Bob is older than Carol, who is the youngest?"
    ]
}


pairs = {
    k: [ 
        (
            strategy, 
            prompt, 
            tokenizer(
                q,
                return_tensors='pt',
                add_special_tokens=False
            ).to(model.device)
        ) 
        for prompt in prompt_list for strategy in strategies for q in queries] for k,queries in query_set.items() 
}

print("Len of pairs:", {k: len(v) for k,v in pairs.items()})
print("Generating prompts for all queries...")

out = { 
    k: [
        gen_prompt_from_query(
            model, 
            tokenizer, 
            query['input_ids'], 
            query['attention_mask'], 
            max_new_tokens=256, 
            instruction=prompt_inst.format(strategy=strat)
        ) 
        for strat,prompt_inst,query in vals 
    ] for k,vals in pairs.items() 
}

print("Generated prompts for all queries")

# [strat_index, {input_ids[0]}] -> {input_ids[strat_index]}
new_out = {k: batch_prompts(val,tokenizer.pad_token_id) for k,val in out.items()}
prompt = {k: [tokenizer.decode(new_out['input_ids'][i], skip_special_tokens=True) for i in range(len(vals))] for k,vals in new_out.items()}

with open('PromptGenerator/query_prompts.txt', 'w') as f:
    for k in prompt.keys():
        f.write(f"{k}:\n")
        for gp, (s,p,qt) in zip(prompt[k], pairs[k]):
            q = tokenizer.decode(qt['input_ids'][0], skip_special_tokens=True)
            f.write(f"Query - {q}\nStrategy - {s}\nInstruct Prompt - {p}\nGenerated Prompt - {gp}\n")
        f.write("\n")
print("Prompts generated and saved to 'PromptGenerator/query_prompts.txt'")

