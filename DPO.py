from accelerate import init_empty_weights, infer_auto_device_map
from transformers import AutoTokenizer, AutoModelForCausalLM
from PreferenceDataLoader import PreferenceDataLoader
import torch
from logger import logger

class DirectPreferenceOptimization:
    def __init__(self, BETA, DEVICE='cuda', LR=1e-5, MAX_LEN=512):
        self.beta = BETA
        self.device = DEVICE
        self.lr = LR
        self.max_len = MAX_LEN

    def set_models(self, MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=self.device)
        self.ref_model.eval()  # Reference model should be in eval mode
        for param in self.ref_model.parameters():
            param.requires_grad = False  # Freeze reference model

        self.policy_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=self.device)
        self.policy_model.train()  # Policy model should be in train mode
        self.policy_optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=self.lr) 
        torch.autograd.set_detect_anomaly(True)

    def collate_fn(self, batch):
        prompt_inputs = self.tokenizer([b['prompt'] for b in batch], return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
        chosen_outputs = self.tokenizer([b['chosen'] for b in batch], return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
        rejected_outputs = self.tokenizer([b['rejected'] for b in batch], return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
        # labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
        return {
            "prompt_inputs": {k: v.to(self.device) for k, v in prompt_inputs.items()},
            "chosen_outputs": {k: v.to(self.device) for k, v in chosen_outputs.items()},
            "rejected_outputs": {k: v.to(self.device) for k, v in rejected_outputs.items()},
            # "labels": labels.to(device)
        }

    def compute_log_prob(self, model, prompt_ids, output_ids, prompt_mask=None, grad=True):
        if not grad:
            torch.set_grad_enabled(False)

        input_ids = torch.cat([prompt_ids, output_ids[:, 1:]], dim=1)
        if prompt_mask is not None:
            attention_mask = torch.cat([prompt_mask, torch.ones_like(output_ids[:, 1:])], dim=1)
        else:
            attention_mask = torch.ones_like(input_ids)

        labels = input_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -100

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        # Per-token loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()
        losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        losses = losses.view(logits.size(0), -1)
        token_mask = (labels != -100)
        log_probs = -torch.sum(losses * token_mask, dim=1)

        if not grad:
            torch.set_grad_enabled(True)
        return log_probs


    def dpo_loss(self, prompt_inputs, chosen_outputs, rejected_outputs):
        prompt_ids = prompt_inputs['input_ids']
        prompt_mask = prompt_inputs['attention_mask']

        # Compute log P(output | prompt) by concatenating prompt + output
        logp_chosen_policy = self.compute_log_prob(self.policy_model, prompt_ids, chosen_outputs['input_ids'], prompt_mask)
        logp_rejected_policy = self.compute_log_prob(self.policy_model, prompt_ids, rejected_outputs['input_ids'], prompt_mask)

        logp_chosen_ref = self.compute_log_prob(self.ref_model, prompt_ids, chosen_outputs['input_ids'], prompt_mask, grad=False)
        logp_rejected_ref = self.compute_log_prob(self.ref_model, prompt_ids, rejected_outputs['input_ids'], prompt_mask, grad=False)

        logp_chosen = logp_chosen_policy - logp_chosen_ref
        logp_rejected = logp_rejected_policy - logp_rejected_ref

        pref_diff = logp_chosen - logp_rejected
        dpo = -torch.nn.functional.logsigmoid(self.beta * pref_diff)
        return dpo.mean()

    def test_model_capability(self,dataloader: PreferenceDataLoader, strategy: str):
        TEST_QUESTION = "Why is 49 not prime?"
        prompted_question = dataloader.build_prompt(TEST_QUESTION, strategy)
        inputs = self.tokenizer(prompted_question, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.policy_model.generate(**inputs, max_new_tokens=self.max_len, do_sample=True)
            output_answer = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        logger.info("### --- Testing --- ###")
        logger.info(f"Test question: {prompted_question}")
        logger.info(f"Test answer: {output_answer}")
        logger.info("### --- End Testing --- ###")

        print("### --- Testing --- ###")
        print(f"Test question: {prompted_question}")
        print(f"Test answer: {output_answer}")
        print("### --- End Testing --- ###")
