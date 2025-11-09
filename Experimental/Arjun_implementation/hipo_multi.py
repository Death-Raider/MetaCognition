import os
import gc
import json
import tqdm
import torch
import torch.nn as nn
from datasets import Dataset
from functools import partial
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


class ConversationDataset:
    """Dataset class for loading and processing JSONL conversation data for HiPO."""
    
    def __init__(self, jsonl_file):
        self.jsonl_file = jsonl_file
        self.conversations = []
        self._load_data()
    
    def _load_data(self):
        """Load and process JSONL file."""
        print(f"Loading data from {self.jsonl_file}...")
        
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    self._process_entry(data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_num} due to JSON error: {e}")
                    continue
        
        print(f"Loaded {len(self.conversations)} conversation pairs")
    
    def _process_entry(self, data):
        """Process a single JSONL entry and extract conversation pairs for HiPO."""
        query = data.get('query', '').strip()
        
        if not query:
            return

        rq_a = data.get('Rq_a', '').strip()
        mt_a = data.get('Mt_a', '').strip()
        ra_a = data.get('Ra_a', '').strip()

        rq_b = data.get('Rq_b', '').strip()
        mt_b = data.get('Mt_b', '').strip()
        ra_b = data.get('Ra_b', '').strip()

        # Extract preferred and rejected components
        preferred = {
            'refined_query': rq_a,
            'meta_thinking': mt_a,
            'refined_answer': ra_a
        }
        
        rejected = {
            'refined_query': rq_b,
            'meta_thinking': mt_b,
            'refined_answer': ra_b
        }
        
        # Skip if any component is empty
        if not all(preferred.values()) or not all(rejected.values()):
            return
        
        # Create full responses by concatenating components
        chosen_response = f"{preferred['refined_query']} {preferred['meta_thinking']} {preferred['refined_answer']}"
        rejected_response = f"{rejected['refined_query']} {rejected['meta_thinking']} {rejected['refined_answer']}"
        
        if chosen_response == rejected_response:
            return 
    
        conversation = {
            "prompt": self._format_prompt(query),
            "chosen": chosen_response,
            "rejected": rejected_response,
            "preferred_components": preferred,
            "rejected_components": rejected,
            "original_query": query
        }
        self.conversations.append(conversation)

    def _format_prompt(self, query):
        """Format prompt using Qwen2.5 chat template."""
        return f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    
    def to_hf_dataset(self):
        """Convert to HuggingFace Dataset."""
        return Dataset.from_list(self.conversations)
    
    def get_sample(self, index=0):
        """Get a sample conversation for inspection."""
        if self.conversations:
            sample = self.conversations[index]
            return {
                "prompt": sample['prompt'],
                "chosen": sample['chosen'],
                "rejected": sample['rejected'],
                "components": sample['preferred_components']
            }
        return None
    
    def __len__(self):
        return len(self.conversations)


def collate_function(data, tokenizer, max_length, device):

    prompts = [item['prompt'] for item in data]
    chosen_responses = [item['chosen'] for item in data]
    rejected_responses = [item['rejected'] for item in data]

    chosen_rqs = [item['preferred_components']['refined_query'] for item in data]
    chosen_mts = [item['preferred_components']['meta_thinking'] for item in data]
    chosen_ras = [item['preferred_components']['refined_answer'] for item in data]

    rejected_rqs = [item['rejected_components']['refined_query'] for item in data]
    rejected_mts = [item['rejected_components']['meta_thinking'] for item in data]
    rejected_ras = [item['rejected_components']['refined_answer'] for item in data]

    chosen_rqs_context = [prompt + " " + rq for prompt, rq in zip(prompts, chosen_rqs)]
    rejected_rqs_context = [prompt + " " + rq for prompt, rq in zip(prompts, rejected_rqs)]

    chosen_mts_context = [prompt + " " + rq + " " + mt for prompt, rq, mt in zip(prompts, chosen_rqs, chosen_mts)]
    rejected_mts_context = [prompt + " " + rq + " " + mt for prompt, rq, mt in zip(prompts, rejected_rqs, rejected_mts)]

    chosen_ras_context = [prompt + " " + rq + " " + mt + " " + ra for prompt, rq, mt, ra in zip(prompts, chosen_rqs, chosen_mts, chosen_ras)]
    rejected_ras_context = [prompt + " " + rq + " " + mt + " " + ra for prompt, rq, mt, ra in zip(prompts, rejected_rqs, rejected_mts, rejected_ras)]


    prompt_encoding = tokenizer(prompts, padding="max_length", truncation=True, max_length=max_length, return_tensors='pt')
    
    full_chosen_responses = [prompt + " " + resp for prompt, resp in zip(prompts, chosen_responses)]
    full_rejected_responses = [prompt + " " + resp for prompt, resp in zip(prompts, rejected_responses)]
    chosen_encoding = tokenizer(full_chosen_responses, padding="max_length", truncation=True, max_length=max_length, return_tensors='pt')
    rejected_encoding = tokenizer(full_rejected_responses, padding="max_length", truncation=True, max_length=max_length, return_tensors='pt')


    rq_a_encoding = tokenizer(chosen_rqs_context, padding="max_length", truncation=True, max_length=max_length, return_tensors='pt')
    mt_a_encoding = tokenizer(chosen_mts_context, padding="max_length", truncation=True, max_length=max_length, return_tensors='pt')
    ra_a_encoding = tokenizer(chosen_ras_context, padding="max_length", truncation=True, max_length=max_length, return_tensors='pt')

    rq_b_encoding = tokenizer(rejected_rqs_context, padding="max_length", truncation=True, max_length=max_length, return_tensors='pt')
    mt_b_encoding = tokenizer(rejected_mts_context, padding="max_length", truncation=True, max_length=max_length, return_tensors='pt')
    ra_b_encoding = tokenizer(rejected_ras_context, padding="max_length", truncation=True, max_length=max_length, return_tensors='pt')

    prompt_length = prompt_encoding.attention_mask.sum(dim=-1).to(device)
    rq_context_length = prompt_length
    mt_a_length = rq_a_encoding.attention_mask.sum(dim=-1).to(device)
    mt_b_length = rq_b_encoding.attention_mask.sum(dim=-1).to(device)
    ra_a_length = mt_a_encoding.attention_mask.sum(dim=-1).to(device)
    ra_b_length = mt_b_encoding.attention_mask.sum(dim=-1).to(device)


    return {
        'prompt_chosen_ids': chosen_encoding.input_ids.to(device),
        'prompt_chosen_mask': chosen_encoding.attention_mask.to(device),
        'prompt_rejected_ids': rejected_encoding.input_ids.to(device),
        'prompt_rejected_mask': rejected_encoding.attention_mask.to(device),
        'prompt_length': prompt_length.to(device),

        'prompt_chosen_rq_ids': rq_a_encoding.input_ids.to(device),
        'prompt_chosen_rq_mask': rq_a_encoding.attention_mask.to(device),
        'prompt_rejected_rq_ids': rq_b_encoding.input_ids.to(device),
        'prompt_rejected_rq_mask': rq_b_encoding.attention_mask.to(device),
        'rq_context_length': rq_context_length,

        'prompt_chosen_mt_ids': mt_a_encoding.input_ids.to(device),
        'prompt_chosen_mt_mask': mt_a_encoding.attention_mask.to(device),
        'prompt_rejected_mt_ids': mt_b_encoding.input_ids.to(device),
        'prompt_rejected_mt_mask': mt_b_encoding.attention_mask.to(device),
        'mt_chosen_context_length' : mt_a_length,
        'mt_rejected_context_length' : mt_b_length,

        'prompt_chosen_ra_ids': ra_a_encoding.input_ids.to(device),
        'prompt_chosen_ra_mask': ra_a_encoding.attention_mask.to(device),
        'prompt_rejected_ra_ids': ra_b_encoding.input_ids.to(device),
        'prompt_rejected_ra_mask': ra_b_encoding.attention_mask.to(device),
        'ra_chosen_context_length' : ra_a_length,
        'ra_rejected_context_length' : ra_b_length,
    }


def get_log_probs(model_logits, labels, tokenizer, prompt_length, normalization = True):
    shift_logits = model_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    shift_labels = shift_labels.masked_fill(shift_labels == tokenizer.pad_token_id, -100)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    # print(log_probs.shape, log_probs, labels)
    token_log_probs = torch.gather(input=log_probs, dim=-1, index=shift_labels.unsqueeze(-1).clamp(min=0)).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(shift_labels == -100, 0.0)
    # print(token_log_probs)

    batch_size, seq_len = shift_labels.shape
    response_mask = (torch.arange(seq_len, device = shift_labels.device).unsqueeze(0) >= prompt_length.unsqueeze(1)).float()
    # print(response_mask)

    response_length = response_mask.sum(dim=-1)
    response_length = torch.clamp(response_length, min=1)
    
    response_log_probs = (token_log_probs * response_mask).sum(dim=-1)
    if normalization:
        return response_log_probs / response_length
    else:
        return response_log_probs


def dpo_loss(model_logprobs_chosen, model_logprobs_rejected, ref_model_logprobs_chosen, ref_model_logprobs_rejected, beta=0.1):
    chosen_logprob = model_logprobs_chosen - ref_model_logprobs_chosen
    rejected_logprob = model_logprobs_rejected - ref_model_logprobs_rejected

    logits = beta * (chosen_logprob - rejected_logprob)
    logits = torch.clamp(logits, min=-50, max=50)

    loss = -F.logsigmoid(logits).mean()
    return loss

def segment_dpo_loss(model, ref_model, batch_data, segment_name,tokenizer, beta=0.1):
    chosen_ids = batch_data[f"prompt_chosen_{segment_name}_ids"]
    chosen_mask = batch_data[f"prompt_chosen_{segment_name}_mask"]
    rejected_ids = batch_data[f"prompt_rejected_{segment_name}_ids"]
    rejected_mask = batch_data[f"prompt_rejected_{segment_name}_mask"]

    if segment_name == 'rq':
        context_length_chosen = batch_data['rq_context_length']
        context_length_rejected = batch_data['rq_context_length']
    elif segment_name == 'mt':
        context_length_chosen = batch_data['mt_chosen_context_length']
        context_length_rejected = batch_data['mt_rejected_context_length']
    elif segment_name == 'ra':
        context_length_chosen = batch_data['ra_chosen_context_length']
        context_length_rejected = batch_data['ra_rejected_context_length']

    model_logits_chosen = model(input_ids = chosen_ids, attention_mask = chosen_mask).logits
    model_logprobs_chosen = get_log_probs(model_logits_chosen, chosen_ids, tokenizer, context_length_chosen)
    del model_logits_chosen

    model_logits_rejected = model(input_ids = rejected_ids, attention_mask = rejected_mask).logits
    model_logprobs_rejected = get_log_probs(model_logits_rejected, rejected_ids, tokenizer, context_length_rejected)
    del model_logits_rejected

    with torch.no_grad():
        ref_model_logits_chosen = ref_model(input_ids = chosen_ids, attention_mask = chosen_mask).logits
        ref_model_logprobs_chosen = get_log_probs(ref_model_logits_chosen, chosen_ids, tokenizer, context_length_chosen)
        del ref_model_logits_chosen

        ref_model_logits_rejected = ref_model(input_ids = rejected_ids, attention_mask = rejected_mask).logits
        ref_model_logprobs_rejected = get_log_probs(ref_model_logits_rejected, rejected_ids, tokenizer, context_length_rejected)
        del ref_model_logits_rejected

    segment_loss = dpo_loss(model_logprobs_chosen, model_logprobs_rejected, ref_model_logprobs_chosen, ref_model_logprobs_rejected, beta)
    return segment_loss

def clear_memory():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

data = ConversationDataset(jsonl_file="semi_automated_dataset_creation/processed_decomposed_dataset.jsonl")
dataset = data.to_hf_dataset()

device = 'cuda'
max_length = 256
batch_size = 1
beta=0.1
epochs = 5
aux_weights = {'rq': 0.2, 'mt': 0.4, 'ra': 0.3}

MODEL_NAME = "/projects/p32722/Models/Qwen2.5-3B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model =  AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device)
ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device)
ref_model.requires_grad_(False)

optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
collater = partial(collate_function, tokenizer = tokenizer, max_length = max_length, device = device)
train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collater)

# make sure tokenizer has a pad token
if tokenizer.pad_token_id is None:
    # set pad token to eos if not present
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        # fallback to 0
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # if you add tokens, you may need to resize model embeddings:
        model.resize_token_embeddings(len(tokenizer))
        ref_model.resize_token_embeddings(len(tokenizer))

for epoch in range(epochs):
    total_epoch_loss = 0
    for i in tqdm.tqdm(train_dataloader):
        optim.zero_grad()

        model_logits_chosen = model(
            input_ids = i['prompt_chosen_ids'],
            attention_mask = i['prompt_chosen_mask']
        ).logits

        model_logprobs_chosen = get_log_probs(
            model_logits = model_logits_chosen,
            labels = i['prompt_chosen_ids'],
            tokenizer = tokenizer,
            prompt_length = i['prompt_length'],
        )
        # print(model_logprobs_chosen)
        del model_logits_chosen
        clear_memory()

        model_logits_rejected = model(
            input_ids = i['prompt_rejected_ids'],
            attention_mask = i['prompt_rejected_mask']
        ).logits

        model_logprobs_rejected = get_log_probs(
            model_logits = model_logits_rejected,
            labels = i['prompt_rejected_ids'],
            tokenizer = tokenizer,
            prompt_length = i['prompt_length']
        )
        # print(model_logprobs_rejected)
        del model_logits_rejected
        clear_memory()

        with torch.no_grad():
            ref_model_logits_chosen = ref_model(
                input_ids = i['prompt_chosen_ids'],
                attention_mask = i['prompt_chosen_mask']
            ).logits

            ref_model_logprobs_chosen = get_log_probs(
                model_logits = ref_model_logits_chosen,
                labels = i['prompt_chosen_ids'],
                tokenizer = tokenizer,
                prompt_length = i['prompt_length']
            )
            # print(ref_model_logprobs_chosen)
            del ref_model_logits_chosen

            ref_model_logits_rejected = ref_model(
                input_ids = i['prompt_rejected_ids'],
                attention_mask = i['prompt_rejected_mask']
            ).logits

            ref_model_logprobs_rejected = get_log_probs(
                model_logits = ref_model_logits_rejected,
                labels = i['prompt_rejected_ids'],
                tokenizer = tokenizer,
                prompt_length = i['prompt_length']
            )
            # print(ref_model_logprobs_rejected)
            del ref_model_logits_rejected


        loss = dpo_loss(
            model_logprobs_chosen,
            model_logprobs_rejected,
            ref_model_logprobs_chosen,
            ref_model_logprobs_rejected,
            beta=beta
        )
        # print(loss)

        rq_loss = segment_dpo_loss(model, ref_model, i, 'rq', tokenizer, beta)
        mt_loss = segment_dpo_loss(model, ref_model, i, 'mt', tokenizer, beta)
        ra_loss = segment_dpo_loss(model, ref_model, i, 'ra', tokenizer, beta)

        auxiliary_loss = aux_weights['rq'] * rq_loss + aux_weights['mt'] * mt_loss + aux_weights['ra'] * ra_loss

        # print(rq_loss, mt_loss, ra_loss)
        total_loss = (0.5 * loss) + (auxiliary_loss)
        # print(total_loss)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        total_epoch_loss += total_loss.item()
     
    print(f"--- Epoch {epoch+1} Average Loss: {total_epoch_loss / len(train_dataloader):.4f} ---")

    # # --- ADDED: CHECKPOINTING AFTER EACH EPOCH ---
    # print(f"--- Checkpointing model after Epoch {epoch+1}... ---")
    # checkpoint_dir = f"checkpoints/epoch_{epoch+1}"
    # os.makedirs(checkpoint_dir, exist_ok=True)
    
    # model.save_pretrained(checkpoint_dir)
    # tokenizer.save_pretrained(checkpoint_dir)
    
    # print(f"Model and tokenizer for Epoch {epoch+1} saved to {checkpoint_dir}")


# --- ADDED: SAVE FINAL MODEL AFTER TRAINING ---
print("\n--- Training finished. Saving final model... ---")
final_model_dir = "final_model_multi_v1"
os.makedirs(final_model_dir, exist_ok=True)

model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

print(f"Final model saved to {final_model_dir}")