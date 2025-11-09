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

    # Create all sequence combinations
    sequences = []
    sequence_metadata = []
    
    for i in range(len(prompts)):
        # Full responses
        sequences.extend([
            prompts[i] + " " + chosen_responses[i],  # chosen full
            prompts[i] + " " + rejected_responses[i]  # rejected full
        ])
        
        # Component sequences  
        sequences.extend([
            prompts[i] + " " + chosen_rqs[i],  # chosen rq
            prompts[i] + " " + rejected_rqs[i],  # rejected rq
            prompts[i] + " " + chosen_rqs[i] + " " + chosen_mts[i],  # chosen mt
            prompts[i] + " " + rejected_rqs[i] + " " + rejected_mts[i],  # rejected mt
            prompts[i] + " " + chosen_rqs[i] + " " + chosen_mts[i] + " " + chosen_ras[i],  # chosen ra
            prompts[i] + " " + rejected_rqs[i] + " " + rejected_mts[i] + " " + rejected_ras[i]  # rejected ra
        ])
        
        # Store metadata for each batch item
        prompt_length = len(tokenizer(prompts[i], add_special_tokens=False)['input_ids'])
        rq_chosen_length = len(tokenizer(prompts[i] + " " + chosen_rqs[i], add_special_tokens=False)['input_ids'])
        rq_rejected_length = len(tokenizer(prompts[i] + " " + rejected_rqs[i], add_special_tokens=False)['input_ids'])
        mt_chosen_length = len(tokenizer(prompts[i] + " " + chosen_rqs[i] + " " + chosen_mts[i], add_special_tokens=False)['input_ids'])
        mt_rejected_length = len(tokenizer(prompts[i] + " " + rejected_rqs[i] + " " + rejected_mts[i], add_special_tokens=False)['input_ids'])
        
        sequence_metadata.append({
            'prompt_length': prompt_length,
            'rq_context_length': prompt_length,
            'mt_chosen_context_length': rq_chosen_length,
            'mt_rejected_context_length': rq_rejected_length,
            'ra_chosen_context_length': mt_chosen_length,
            'ra_rejected_context_length': mt_rejected_length,
        })
    
    # Tokenize all sequences at once
    all_encodings = tokenizer(sequences, padding="max_length", truncation=True, 
                             max_length=max_length, return_tensors='pt')
    
    batch_size = len(prompts)
    sequences_per_item = 8  # 2 full + 6 component sequences
    
    # Reshape into batch format
    batch_data = {}
    for i in range(batch_size):
        base_idx = i * sequences_per_item
        
        batch_data[f'item_{i}'] = {
            # Full sequences (indices 0, 1)
            'full_chosen_ids': all_encodings.input_ids[base_idx].to(device),
            'full_chosen_mask': all_encodings.attention_mask[base_idx].to(device),
            'full_rejected_ids': all_encodings.input_ids[base_idx + 1].to(device),
            'full_rejected_mask': all_encodings.attention_mask[base_idx + 1].to(device),
            
            # Component sequences (indices 2-7)
            'rq_chosen_ids': all_encodings.input_ids[base_idx + 2].to(device),
            'rq_chosen_mask': all_encodings.attention_mask[base_idx + 2].to(device),
            'rq_rejected_ids': all_encodings.input_ids[base_idx + 3].to(device),
            'rq_rejected_mask': all_encodings.attention_mask[base_idx + 3].to(device),
            
            'mt_chosen_ids': all_encodings.input_ids[base_idx + 4].to(device),
            'mt_chosen_mask': all_encodings.attention_mask[base_idx + 4].to(device),
            'mt_rejected_ids': all_encodings.input_ids[base_idx + 5].to(device),
            'mt_rejected_mask': all_encodings.attention_mask[base_idx + 5].to(device),
            
            'ra_chosen_ids': all_encodings.input_ids[base_idx + 6].to(device),
            'ra_chosen_mask': all_encodings.attention_mask[base_idx + 6].to(device),
            'ra_rejected_ids': all_encodings.input_ids[base_idx + 7].to(device),
            'ra_rejected_mask': all_encodings.attention_mask[base_idx + 7].to(device),
            
            **sequence_metadata[i]
        }
    
    return {
        'batch_size': batch_size,
        'all_input_ids': all_encodings.input_ids.to(device),
        'all_attention_mask': all_encodings.attention_mask.to(device),
        'sequences_per_item': sequences_per_item,
        'metadata': sequence_metadata
    }


def get_log_probs(model_logits, labels, tokenizer, prompt_length, normalization=True):
    shift_logits = model_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_labels = shift_labels.masked_fill(shift_labels == tokenizer.pad_token_id, -100)
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(input=log_probs, dim=-1, index=shift_labels.unsqueeze(-1).clamp(min=0)).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(shift_labels == -100, 0.0)
    
    batch_size, seq_len = shift_labels.shape
    response_mask = (torch.arange(seq_len, device=shift_labels.device).unsqueeze(0) >= prompt_length.unsqueeze(1)).float()
    
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


def compute_all_losses(model_logits, ref_logits, batch_data, tokenizer, beta=0.1):
    """Compute main and auxiliary losses from batched forward pass results."""
    batch_size = batch_data['batch_size']
    sequences_per_item = batch_data['sequences_per_item']
    metadata = batch_data['metadata']
    
    total_main_loss = 0
    total_aux_losses = {'rq': 0, 'mt': 0, 'ra': 0}
    
    for i in range(batch_size):
        base_idx = i * sequences_per_item
        meta = metadata[i]
        
        # Extract logits for each sequence type
        full_chosen_logits = model_logits[base_idx]
        full_rejected_logits = model_logits[base_idx + 1]
        rq_chosen_logits = model_logits[base_idx + 2]
        rq_rejected_logits = model_logits[base_idx + 3]
        mt_chosen_logits = model_logits[base_idx + 4]
        mt_rejected_logits = model_logits[base_idx + 5]
        ra_chosen_logits = model_logits[base_idx + 6]
        ra_rejected_logits = model_logits[base_idx + 7]
        
        # Same for reference model
        ref_full_chosen_logits = ref_logits[base_idx]
        ref_full_rejected_logits = ref_logits[base_idx + 1]
        ref_rq_chosen_logits = ref_logits[base_idx + 2]
        ref_rq_rejected_logits = ref_logits[base_idx + 3]
        ref_mt_chosen_logits = ref_logits[base_idx + 4]
        ref_mt_rejected_logits = ref_logits[base_idx + 5]
        ref_ra_chosen_logits = ref_logits[base_idx + 6]
        ref_ra_rejected_logits = ref_logits[base_idx + 7]
        
        # Get corresponding input_ids and labels
        all_ids = batch_data['all_input_ids']
        full_chosen_ids = all_ids[base_idx].unsqueeze(0)
        full_rejected_ids = all_ids[base_idx + 1].unsqueeze(0)
        rq_chosen_ids = all_ids[base_idx + 2].unsqueeze(0)
        rq_rejected_ids = all_ids[base_idx + 3].unsqueeze(0)
        mt_chosen_ids = all_ids[base_idx + 4].unsqueeze(0)
        mt_rejected_ids = all_ids[base_idx + 5].unsqueeze(0)
        ra_chosen_ids = all_ids[base_idx + 6].unsqueeze(0)
        ra_rejected_ids = all_ids[base_idx + 7].unsqueeze(0)
        
        # Compute log probabilities for main loss
        prompt_len_tensor = torch.tensor([meta['prompt_length']], device=model_logits.device)
        
        main_chosen_logprobs = get_log_probs(full_chosen_logits.unsqueeze(0), full_chosen_ids, tokenizer, prompt_len_tensor)
        main_rejected_logprobs = get_log_probs(full_rejected_logits.unsqueeze(0), full_rejected_ids, tokenizer, prompt_len_tensor)
        ref_main_chosen_logprobs = get_log_probs(ref_full_chosen_logits.unsqueeze(0), full_chosen_ids, tokenizer, prompt_len_tensor)
        ref_main_rejected_logprobs = get_log_probs(ref_full_rejected_logits.unsqueeze(0), full_rejected_ids, tokenizer, prompt_len_tensor)
        
        # Main DPO loss
        main_loss = dpo_loss(main_chosen_logprobs, main_rejected_logprobs, 
                           ref_main_chosen_logprobs, ref_main_rejected_logprobs, beta)
        total_main_loss += main_loss
        
        # Auxiliary losses
        # RQ loss
        rq_context_len_tensor = torch.tensor([meta['rq_context_length']], device=model_logits.device)
        rq_chosen_logprobs = get_log_probs(rq_chosen_logits.unsqueeze(0), rq_chosen_ids, tokenizer, rq_context_len_tensor)
        rq_rejected_logprobs = get_log_probs(rq_rejected_logits.unsqueeze(0), rq_rejected_ids, tokenizer, rq_context_len_tensor)
        ref_rq_chosen_logprobs = get_log_probs(ref_rq_chosen_logits.unsqueeze(0), rq_chosen_ids, tokenizer, rq_context_len_tensor)
        ref_rq_rejected_logprobs = get_log_probs(ref_rq_rejected_logits.unsqueeze(0), rq_rejected_ids, tokenizer, rq_context_len_tensor)
        
        rq_loss = dpo_loss(rq_chosen_logprobs, rq_rejected_logprobs, ref_rq_chosen_logprobs, ref_rq_rejected_logprobs, beta)
        total_aux_losses['rq'] += rq_loss
        
        # MT loss
        mt_chosen_context_len_tensor = torch.tensor([meta['mt_chosen_context_length']], device=model_logits.device)
        mt_rejected_context_len_tensor = torch.tensor([meta['mt_rejected_context_length']], device=model_logits.device)
        
        mt_chosen_logprobs = get_log_probs(mt_chosen_logits.unsqueeze(0), mt_chosen_ids, tokenizer, mt_chosen_context_len_tensor)
        mt_rejected_logprobs = get_log_probs(mt_rejected_logits.unsqueeze(0), mt_rejected_ids, tokenizer, mt_rejected_context_len_tensor)
        ref_mt_chosen_logprobs = get_log_probs(ref_mt_chosen_logits.unsqueeze(0), mt_chosen_ids, tokenizer, mt_chosen_context_len_tensor)
        ref_mt_rejected_logprobs = get_log_probs(ref_mt_rejected_logits.unsqueeze(0), mt_rejected_ids, tokenizer, mt_rejected_context_len_tensor)
        
        mt_loss = dpo_loss(mt_chosen_logprobs, mt_rejected_logprobs, ref_mt_chosen_logprobs, ref_mt_rejected_logprobs, beta)
        total_aux_losses['mt'] += mt_loss
        
        # RA loss
        ra_chosen_context_len_tensor = torch.tensor([meta['ra_chosen_context_length']], device=model_logits.device)
        ra_rejected_context_len_tensor = torch.tensor([meta['ra_rejected_context_length']], device=model_logits.device)
        
        ra_chosen_logprobs = get_log_probs(ra_chosen_logits.unsqueeze(0), ra_chosen_ids, tokenizer, ra_chosen_context_len_tensor)
        ra_rejected_logprobs = get_log_probs(ra_rejected_logits.unsqueeze(0), ra_rejected_ids, tokenizer, ra_rejected_context_len_tensor)
        ref_ra_chosen_logprobs = get_log_probs(ref_ra_chosen_logits.unsqueeze(0), ra_chosen_ids, tokenizer, ra_chosen_context_len_tensor)
        ref_ra_rejected_logprobs = get_log_probs(ref_ra_rejected_logits.unsqueeze(0), ra_rejected_ids, tokenizer, ra_rejected_context_len_tensor)
        
        ra_loss = dpo_loss(ra_chosen_logprobs, ra_rejected_logprobs, ref_ra_chosen_logprobs, ref_ra_rejected_logprobs, beta)
        total_aux_losses['ra'] += ra_loss
    
    # Average losses across batch
    avg_main_loss = total_main_loss / batch_size
    avg_aux_losses = {k: v / batch_size for k, v in total_aux_losses.items()}
    
    return avg_main_loss, avg_aux_losses


def clear_memory():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# Training setup
data = ConversationDataset(jsonl_file="semi_automated_dataset_creation/processed_decomposed_dataset.jsonl")
dataset = data.to_hf_dataset()

device = 'cuda'
max_length = 256
batch_size = 2
beta = 0.1
epochs = 5
lr = 8e-6
aux_weights = {'rq': 0.2, 'mt': 0.5, 'ra': 0.2}

MODEL_NAME = "/projects/p32722/Models/Qwen2.5-3B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device)
ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device)
ref_model.requires_grad_(False)

optim = torch.optim.AdamW(model.parameters(), lr=lr)
collater = partial(collate_function, tokenizer=tokenizer, max_length=max_length, device=device)
train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collater)

# # Handle tokenizer pad token
# if tokenizer.pad_token_id is None:
#     if tokenizer.eos_token_id is not None:
#         tokenizer.pad_token = tokenizer.eos_token
#     else:
#         tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#         model.resize_token_embeddings(len(tokenizer))
#         ref_model.resize_token_embeddings(len(tokenizer))

# Training loop
for epoch in range(epochs):
    print(f"Epoch: {epoch + 1}")
    total_epoch_loss = 0
    for batch_data in tqdm.tqdm(train_dataloader):
        optim.zero_grad()
        
        # Single forward pass for all sequences
        model_logits = model(
            input_ids=batch_data['all_input_ids'],
            attention_mask=batch_data['all_attention_mask']
        ).logits
        
        # Single forward pass for reference model
        with torch.no_grad():
            ref_logits = ref_model(
                input_ids=batch_data['all_input_ids'],
                attention_mask=batch_data['all_attention_mask']
            ).logits
        
        # Compute all losses from the batched results
        main_loss, aux_losses = compute_all_losses(model_logits, ref_logits, batch_data, tokenizer, beta)
        
        # Combine losses
        auxiliary_loss = sum(aux_weights[k] * aux_losses[k] for k in aux_weights.keys())
        total_loss = 0.10 * main_loss + auxiliary_loss
        
        # print(f"Main: {main_loss:.4f}, RQ: {aux_losses['rq']:.4f}, MT: {aux_losses['mt']:.4f}, RA: {aux_losses['ra']:.4f}, Total: {total_loss:.4f}")
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        
        total_epoch_loss += total_loss.item()
        clear_memory()
     
    print(f"--- Epoch {epoch+1} Average Loss: {total_epoch_loss / len(train_dataloader):.4f} ---")

    # # Checkpointing
    # checkpoint_dir = f"checkpoints/epoch_{epoch+1}"
    # os.makedirs(checkpoint_dir, exist_ok=True)
    # model.save_pretrained(checkpoint_dir)
    # tokenizer.save_pretrained(checkpoint_dir)
    # print(f"Model saved to {checkpoint_dir}")

# Save final model
final_model_dir = "final_model_v3"
os.makedirs(final_model_dir, exist_ok=True)
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
print(f"Final model saved to {final_model_dir}")