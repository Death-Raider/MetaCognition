# ========== imports ==============
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
# from torchviz import make_dot
from PreferenceDataLoader import PreferenceDataLoader
from DPO import DirectPreferenceOptimization
import json
from ConfigSchema import ConfigSchema
from datasets import load_dataset
from logger import logger

# ========== Config Loading ==============
config_schema = ConfigSchema()
with open("config.cfg", "r") as cfg:
    config = {}
    for line in cfg:
        if line.strip() and not line.startswith("#"):
            key, value = line.strip().split("=")
            config[key.strip()] = value.strip()
config_schema.from_dict(config)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Config loaded:", config_schema)
logger.info(f"Config loaded:{config_schema}")
print("Device set as:", DEVICE)
logger.info(f"Device set as:{DEVICE}")
    
# === Example dummy dataset ===
"""
{
  "query": <original query>,
  "output_a": <model output a>,
  "M_a_text": <explanation of meta-cognitive plan>,
  "M_a_span": [M_start, M_end],
  "T_a_text": <explanation of tactical plan>,
  "T_a_span": [T_start, T_end],
  "A_a_text": <verbatim answer span>,
  "A_a_span": [A_start, A_end],
  "S_a": <short strategy label>,
  "output_b": <model output b>,
  "M_b_text": <explanation of meta-cognitive plan>,
  "M_b_span": [M_start, M_end],
  "T_b_text": <explanation of tactical plan>,
  "T_b_span": [T_start, T_end],
  "A_b_text": <verbatim answer span>,
  "A_b_span": [A_start, A_end],
  "S_b": <short strategy label>,
  "label": <original label>
}
"""
with open('semi_automated_dataset_creation/processed_decomposed_dataset.jsonl', 'r') as f:
    preference = [json.loads(line) for line in f]


# ====== Initialize DPO and DataLoader ======
dataset = preference
DPO = DirectPreferenceOptimization(config_schema.beta, DEVICE, config_schema.lr, config_schema.max_len)
DPO.set_models(config_schema.model_name)

def collate_fn(batch):
    prompt_inputs = DPO.tokenizer([b['query'] for b in batch], return_tensors="pt", padding=True, truncation=True, max_length=config_schema.max_len)
    output_a = DPO.tokenizer([b['output_a'] for b in batch], return_tensors="pt", padding=True, truncation=True, max_length=config_schema.max_len)
    output_b = DPO.tokenizer([b['output_b'] for b in batch], return_tensors="pt", padding=True, truncation=True, max_length=config_schema.max_len)
    return {
        "query": {k: v.to(DEVICE) for k, v in prompt_inputs.items()},
        "output_a": {k: v.to(DEVICE) for k, v in output_a.items()},
        "output_b": {k: v.to(DEVICE) for k, v in output_b.items()},
    } | {k:[b[k] for b in batch] for k in batch[0].keys() if k not in ['query', 'output_a', 'output_b']}

loader = DataLoader(dataset, batch_size=config_schema.batch_size, shuffle=True, collate_fn=collate_fn)

# ====== DPO Loss Function ======
# def compute_generation_logprobs(generation_output, input_length):
#     logprobs = []
#     for i, scores in enumerate(generation_output.scores):
#         # Get tokens generated at this step
#         tokens = generation_output.sequences[:, input_length + i]
        
#         # Compute log probabilities
#         log_probs = torch.log_softmax(scores, dim=-1)
#         token_logprobs = log_probs.gather(1, tokens.unsqueeze(-1)).squeeze(-1)
#         logprobs.append(token_logprobs)
    
#     # Sum log probabilities across all generated tokens
#     return torch.stack(logprobs, dim=1).sum(dim=1)

def gen_prompt_from_query(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    max_new_tokens=64,
    gen_prompt_ids=None,
    instruction="Generate a concise reasoning prompt to answer the following question:"
):
    """
    Generates a self-prompt conditioning on the input query and optional base instruction.

    Args:
        model: The LLM model
        input_ids: Tokenized input query [batch_size, seq_len]
        attention_mask: Attention mask for input [batch_size, seq_len]
        max_new_tokens: Maximum length of generated prompt
        gen_prompt_ids: Optional pre-tokenized instruction prompt
        instruction: Text instruction for prompt generation (ignored if gen_prompt_ids provided)

    Returns:
        dict with:
            'input_ids': Generated prompt tokens [batch_size, prompt_len]
            'attention_mask': Mask for generated tokens
            'log_probs': Sum of log probs for generated tokens
            'full_input_ids': (instruction + query) tokens before generation
    """
    with torch.no_grad():
    # Tokenize instruction if not provided
        if gen_prompt_ids is None:
            gen_prompt_ids = tokenizer(
                instruction,
                return_tensors='pt',
                add_special_tokens=False
            ).input_ids.to(input_ids.device)
        logger.info(f"Generated prompt tokenized IDs")
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
        prompt_ids = outputs.sequences

        return {
            "input_ids": prompt_ids,
            "attention_mask": torch.ones_like(prompt_ids),
            # "log_probs": compute_generation_logprobs(outputs, full_input_ids.size(1)),
            "full_input_ids": full_input_ids,
        }

def compute_log_prob_spans(model, input_ids, input_mask, output_ids, spans: list[list[int]], grad=True):
    """
    Compute log probabilities for spans in the output sequence
    Returns: (total_log_probs, [M_log_probs, T_log_probs, A_log_probs])
    """
    with torch.set_grad_enabled(grad):
        # Correct concatenation: input + output WITHOUT last token
        inputs = torch.cat([input_ids, output_ids[:, :-1]], dim=1)
        
        # Create attention mask
        if input_mask is not None:
            output_mask = torch.ones_like(output_ids[:, :-1])
            attention_mask = torch.cat([input_mask, output_mask], dim=1)
        else:
            attention_mask = torch.ones_like(inputs)
        
        # Labels: -100 for input, actual tokens for output
        labels = torch.full_like(inputs, -100)
        labels[:, input_ids.size(1):] = output_ids[:, 1:]
        
        # Forward pass
        outputs = model(
            input_ids=inputs,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Get per-token losses
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        ).view(shift_labels.shape)
        
        # Create mask for valid positions
        valid_mask = (shift_labels != -100)
        
        # Total log probability for full output
        total_log_probs = -losses.masked_fill(~valid_mask, 0).sum(dim=1)
        
        # spans: [n, batch, 2]
        span_log_probs = []

        n_spans = len(spans)       # 3
        batch_size = len(spans[0]) # 2

        for i in range(n_spans):  # For each span type
            # create a fresh mask for all batch items
            span_mask = torch.zeros_like(valid_mask, dtype=torch.bool)

            for b in range(batch_size):
                start, end = spans[i][b]
                # clamp to valid range
                start = max(0, start)
                end = min(valid_mask.shape[1], end)
                span_mask[b, start:end] = valid_mask[b, start:end]

            # compute log prob for this span type across batch
            span_log_probs.append(
                -losses.masked_fill(~span_mask, 0).sum(dim=1)
            )
    return total_log_probs, span_log_probs

def dpo_loss(batch, beta):

    P_a = gen_prompt_from_query(
        model=DPO.policy_model, 
        tokenizer=DPO.tokenizer, 
        input_ids=batch['query']['input_ids'],
        attention_mask=batch['query']['attention_mask'],
        max_new_tokens=config_schema.max_len,
        instruction=f"Generate a prompt to answer the following question using {batch['S_a']} without any extra output. Only answer with the prompt:",
    )
    logger.info(f"Generated prompt for A")

    P_b = gen_prompt_from_query(
        model=DPO.policy_model,
        tokenizer=DPO.tokenizer, 
        input_ids=batch['query']['input_ids'],
        attention_mask=batch['query']['attention_mask'],
        max_new_tokens=config_schema.max_len,
        instruction=f"Generate a prompt to answer the following question using {batch['S_b']} without any extra output. Only answer with the prompt:",
    )
    logger.info(f"Generated prompt for A")

    print("Generated prompts")

    A_log_probs, [A_log_probs_M, A_log_probs_T, A_log_probs_A] = compute_log_prob_spans(
        DPO.policy_model, 
        input_ids=P_a['input_ids'],
        input_mask=P_a['attention_mask'],
        output_ids=batch['output_a']['input_ids'], 
        spans=[ batch['M_a_span'], 
               batch['T_a_span'],
               batch['A_a_span'] ]
    )
    B_log_probs, [B_log_probs_M, B_log_probs_T, B_log_probs_A] = compute_log_prob_spans(
        DPO.policy_model, 
        input_ids=P_b['input_ids'],
        input_mask=P_b['attention_mask'],
        output_ids=batch['output_b']['input_ids'], 
        spans=[ batch['M_b_span'], 
               batch['T_b_span'],
               batch['A_b_span'] ]
    )

    A_log_probs_ref, [A_log_probs_M_ref, A_log_probs_T_ref, A_log_probs_A_ref] = compute_log_prob_spans(
        DPO.ref_model, 
        input_ids=P_a['input_ids'],
        input_mask=P_a['attention_mask'],
        output_ids=batch['output_a']['input_ids'], 
        spans=[ batch['M_a_span'], 
               batch['T_a_span'],
               batch['A_a_span'] ],
        grad=False
    )
    B_log_probs_ref, [B_log_probs_M_ref, B_log_probs_T_ref, B_log_probs_A_ref] = compute_log_prob_spans(
        DPO.ref_model, 
        input_ids=P_b['input_ids'],
        input_mask=P_b['attention_mask'],
        output_ids=batch['output_b']['input_ids'], 
        spans=[ batch['M_b_span'], 
               batch['T_b_span'],
               batch['A_b_span'] ],
        grad=False
    )

    print("Computed log probabilities")
    logger.info("Computed log probabilities")

    # Advantge calculation = Policy - Reference
    A_loss = A_log_probs - A_log_probs_ref
    B_loss = B_log_probs - B_log_probs_ref

    A_loss_M = A_log_probs_M - A_log_probs_M_ref
    A_loss_T = A_log_probs_T - A_log_probs_T_ref
    A_loss_A = A_log_probs_A - A_log_probs_A_ref

    B_loss_M = B_log_probs_M - B_log_probs_M_ref
    B_loss_T = B_log_probs_T - B_log_probs_T_ref
    B_loss_A = B_log_probs_A - B_log_probs_A_ref

    batch_label = torch.as_tensor(batch['label'], device=DEVICE, dtype=torch.float32)
    # sign: -1 if label > 0 else 1  (vectorized)
    sign = torch.where(batch_label > 0, -1.0, 1.0)  # shape: [batch]

    # preferred - dispreferred
    loss_M = (A_loss_M - B_loss_M)*sign
    loss_T = (A_loss_T - B_loss_T)*sign
    loss_A = (A_loss_A - B_loss_A)*sign
    loss_MTAS = (A_loss - B_loss)*sign

    with torch.no_grad():
        var_M = torch.var(torch.stack([A_loss_M, B_loss_M]), dim=0)
        var_T = torch.var(torch.stack([A_loss_T, B_loss_T]), dim=0)
        var_A = torch.var(torch.stack([A_loss_A, B_loss_A]), dim=0)
        var_MTAS = torch.var(torch.stack([A_loss, B_loss]), dim=0)
        total_var = var_M + var_T + var_A + var_MTAS + 1e-8
        w_M = var_M / total_var
        w_T = var_T / total_var
        w_A = var_A / total_var
        w_MTAS = var_MTAS / total_var

    logger.info(f"Loss weights: M={w_M:.4f}, T={w_T:.4f}, A={w_A:.4f}, MTAS={w_MTAS:.4f}")

    strength = torch.abs(batch_label) / 3.0  # Normalized to [0,1]
    loss_M = -strength * torch.nn.functional.logsigmoid(beta * loss_M)
    loss_T = -strength * torch.nn.functional.logsigmoid(beta * loss_T)
    loss_A = -strength * torch.nn.functional.logsigmoid(beta * loss_A)
    loss_MTAS = -strength * torch.nn.functional.logsigmoid(beta * loss_MTAS)

    loss = loss_M*w_M + loss_T*w_T + loss_A*w_A + loss_MTAS*w_MTAS # Loss by M + Loss by T + Loss by A + Loss by complete Trace

    loss = torch.mean(torch.clamp(loss, min=0.0))  # Ensure non-negative loss
    logger.info(f"Loss: {loss.item():.4f}")
    return loss

# ====== Training loop ======
for epoch in range(config_schema.epochs):
    total_loss = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch + 1}"):
        DPO.policy_optimizer.zero_grad()
        loss = dpo_loss(batch, beta = config_schema.beta)
        loss.backward()
        DPO.policy_optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1} Loss: {total_loss / len(loader):.4f}")
    logger.info(f"Epoch {epoch + 1} Loss: {total_loss / len(loader):.4f}")
