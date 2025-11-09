import torch

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