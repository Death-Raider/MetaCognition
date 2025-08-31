from transformers import AutoTokenizer, AutoModelForCausalLM
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
        self.ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(self.device)
        self.ref_model.eval()  # Reference model should be in eval mode
        for param in self.ref_model.parameters():
            param.requires_grad = False  # Freeze reference model

        self.policy_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(self.device)
        self.policy_model.train()  # Policy model should be in train mode
        self.policy_optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=self.lr) 
        torch.autograd.set_detect_anomaly(True)

    def collate_fn(self,batch):
        prompt_inputs = self.tokenizer([b['query'] for b in batch], return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
        # for span calculation
        Rq_a = self.tokenizer([b['Rq_a'] for b in batch], return_tensors="pt", padding=True)
        Mt_a = self.tokenizer([b['Mt_a'] for b in batch], return_tensors="pt", padding=True)
        Ra_a = self.tokenizer([b['Ra_a'] for b in batch], return_tensors="pt", padding=True)
        y_a_lengths = [
            Rq_a["attention_mask"].sum(dim=1).tolist(),
            Mt_a["attention_mask"].sum(dim=1).tolist(),
            Ra_a["attention_mask"].sum(dim=1).tolist()
        ]
        Rq_a_span = [
            (0, y_a_lengths[0][i])
            for i in range(len(batch))
        ]
        Mt_a_span = [
            (y_a_lengths[0][i], y_a_lengths[0][i]+y_a_lengths[1][i])
            for i in range(len(batch))
        ]
        Ra_a_span = [
            ( y_a_lengths[0][i]+y_a_lengths[1][i],  y_a_lengths[0][i]+y_a_lengths[1][i]+y_a_lengths[2][i])
            for i in range(len(batch))
        ]

        Rq_b = self.tokenizer([b['Rq_b'] for b in batch], return_tensors="pt",padding=True)
        Mt_b = self.tokenizer([b['Mt_b'] for b in batch], return_tensors="pt",padding=True)
        Ra_b = self.tokenizer([b['Ra_b'] for b in batch], return_tensors="pt",padding=True)
        y_b_lengths = [
            Rq_b["attention_mask"].sum(dim=1).tolist(),
            Mt_b["attention_mask"].sum(dim=1).tolist(),
            Ra_b["attention_mask"].sum(dim=1).tolist()
        ]
        Rq_b_span = [
            (0, y_b_lengths[0][i])
            for i in range(len(batch))
        ]
        Mt_b_span = [
            (y_b_lengths[0][i], y_b_lengths[0][i]+y_b_lengths[1][i])
            for i in range(len(batch))
        ]
        Ra_b_span = [
            ( y_b_lengths[0][i]+y_b_lengths[1][i],  y_b_lengths[0][i]+y_b_lengths[1][i]+y_b_lengths[2][i])
            for i in range(len(batch))
        ]
        # joined tokeniation might be different from individual tokenization due to truncation/padding
        output_a = self.tokenizer([b['new_output_a'] for b in batch], return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
        output_b = self.tokenizer([b['new_output_b'] for b in batch], return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
        
        return {
            "query": {k: v.to(self.device) for k, v in prompt_inputs.items()},
            "output_a": {k: v.to(self.device) for k, v in output_a.items()},
            "output_b": {k: v.to(self.device) for k, v in output_b.items()},
            "Ra_a_span": Ra_a_span,
            "Mt_a_span": Mt_a_span,
            "Rq_a_span": Rq_a_span,

            "Ra_b_span": Ra_b_span,
            "Mt_b_span": Mt_b_span,
            "Rq_b_span": Rq_b_span
        } | {k:[b[k] for b in batch] for k in batch[0].keys() if k not in ['query', 'output_a', 'output_b']}

    def compute_log_prob_spans(self, model, input_ids, input_mask, output_ids, spans: list[list[int]], grad=True):
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
            labels = inputs.clone()
            labels[:, :input_ids.size(1)] = -100
            
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
            total_log_probs = -(losses*valid_mask).sum(dim=1)

            # spans: [n, batch, 2]
            span_log_probs = []

            n_spans = len(spans)       # 3
            batch_size = len(spans[0]) # 2

            for i in range(n_spans):  # For each span type (M, T, A)
                span_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
                for b in range(batch_size):
                    if len(spans[i][b]) < 2:
                        continue
                    s, e = spans[i][b]
                    
                    # Offset spans by the input length
                    s = input_ids.size(1) + s
                    e = input_ids.size(1) + e

                    # Clamp to avoid out-of-range
                    s = max(s, input_ids.size(1))
                    e = min(e, valid_mask.shape[1])

                    if s < e:
                        span_mask[b, s:e] = True

                span_log_probs.append(
                    -losses.masked_fill(~span_mask, 0).sum(dim=1)
                )
                # print(f"Span {i} log probs: {span_log_probs[-1]}")
            return total_log_probs, span_log_probs
    
    def dpo_loss(self, batch, Prompt_Instruction,beta,weights):
        batch_size = batch['output_a']['input_ids'].size(0)
        new_query = {}
        new_query['input_ids'] = torch.cat([batch['query']['input_ids'], Prompt_Instruction['input_ids'].repeat(batch_size,1)], dim=1)
        new_query['attention_mask'] = torch.cat([batch['query']['attention_mask'], Prompt_Instruction['attention_mask'].repeat(batch_size,1)], dim=1)

        # prompt_text = self.tokenizer.decode(P_b['input_ids'][0], skip_special_tokens=True)
        # logger.info(f"Generated prompt for B:{prompt_text}")

        A_log_probs, [A_log_probs_M, A_log_probs_T, A_log_probs_A] = self.compute_log_prob_spans(
            self.policy_model, 
            input_ids=new_query['input_ids'],
            input_mask=new_query['attention_mask'],
            output_ids=batch['output_a']['input_ids'], 
            spans=[ batch['Rq_a_span'], 
                batch['Mt_a_span'],
                batch['Ra_a_span'] ]
        )
        B_log_probs, [B_log_probs_M, B_log_probs_T, B_log_probs_A] = self.compute_log_prob_spans(
            self.policy_model, 
            input_ids=new_query['input_ids'],
            input_mask=new_query['attention_mask'],
            output_ids=batch['output_b']['input_ids'], 
            spans=[ batch['Rq_b_span'], 
                batch['Mt_b_span'],
                batch['Ra_b_span'] ]
        )

        A_log_probs_ref, [A_log_probs_M_ref, A_log_probs_T_ref, A_log_probs_A_ref] = self.compute_log_prob_spans(
            self.ref_model, 
            input_ids=new_query['input_ids'],
            input_mask=new_query['attention_mask'],
            output_ids=batch['output_a']['input_ids'], 
            spans=[ batch['Rq_a_span'], 
                batch['Mt_a_span'],
                batch['Ra_a_span'] ],
            grad=False
        )
        B_log_probs_ref, [B_log_probs_M_ref, B_log_probs_T_ref, B_log_probs_A_ref] = self.compute_log_prob_spans(
            self.ref_model, 
            input_ids=new_query['input_ids'],
            input_mask=new_query['attention_mask'],
            output_ids=batch['output_b']['input_ids'], 
            spans=[ batch['Rq_b_span'], 
                batch['Mt_b_span'],
                batch['Ra_b_span'] ],
            grad=False
        )

        # Advantge calculation = Policy - Reference
        A_loss = A_log_probs - A_log_probs_ref
        B_loss = B_log_probs - B_log_probs_ref

        A_loss_M = A_log_probs_M - A_log_probs_M_ref
        A_loss_T = A_log_probs_T - A_log_probs_T_ref
        A_loss_A = A_log_probs_A - A_log_probs_A_ref

        B_loss_M = B_log_probs_M - B_log_probs_M_ref
        B_loss_T = B_log_probs_T - B_log_probs_T_ref
        B_loss_A = B_log_probs_A - B_log_probs_A_ref


        # preferred - dispreferred
        loss_M = (A_loss_M - B_loss_M)
        loss_T = (A_loss_T - B_loss_T)
        loss_A = (A_loss_A - B_loss_A)
        loss_MTAS = (A_loss - B_loss)

        # logger.info(f"A_loss_M: {A_loss_M}")
        # logger.info(f"B_loss_M: {B_loss_M}")

        # logger.info(f"A_loss_T: {A_loss_T}")
        # logger.info(f"B_loss_T: {B_loss_T}")

        # logger.info(f"A_loss_A: {A_loss_A}")
        # logger.info(f"B_loss_A: {B_loss_A}")

        w_M = weights[0]
        w_T = weights[1]
        w_A = weights[2]
        w_MTAS = weights[3]

        strength = 1.0
        loss_M = -strength * torch.nn.functional.logsigmoid(beta * loss_M + 0.01)
        loss_T = -strength * torch.nn.functional.logsigmoid(beta * loss_T + 0.01)
        loss_A = -strength * torch.nn.functional.logsigmoid(beta * loss_A + 0.01)
        loss_MTAS = -strength * torch.nn.functional.logsigmoid(beta * loss_MTAS + 0.01)

        loss = loss_M*w_M + loss_T*w_T + loss_A*w_A + loss_MTAS*w_MTAS # Loss by M + Loss by T + Loss by A + Loss by complete Trace

        loss = torch.mean(torch.clamp(loss, min=0.0))  # Ensure non-negative loss
        # logger.info(f"Loss: {loss.item():.4f}")
        return loss
