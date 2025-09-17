import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig


class ConversationDataset:
    """Dataset class for loading and processing JSONL conversation data for DPO."""
    
    def __init__(self, jsonl_file):
        self.jsonl_file = jsonl_file
        self.conversations = []
        self._load_data()
    
    def _load_data(self):
        """Load and process JSONL file."""
        print(f"Loading data from {self.jsonl_file}...")
        
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                self._process_entry(data)
        
        print(f"Loaded {len(self.conversations)} conversation pairs")
    
    def _process_entry(self, data):
        """Process a single JSONL entry and extract conversation pairs for DPO."""
        query = data.get('query', '').strip()
        
        if not query:
            return

        # For DPO, we need chosen and rejected responses
        # Assuming output_a is preferred over output_b (you can modify this logic)
        chosen = data['Rq_a'] + ' ' + data['Mt_a'] + ' ' + data['Ra_a']
        rejected = data['Rq_b'] + ' ' + data['Mt_b'] + ' ' + data['Ra_b']
        
        # Skip if either response is empty
        if not chosen or not rejected:
            return
        
        conversation = {
            "prompt": self._format_prompt(query),
            "chosen": chosen,
            "rejected": rejected
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
            return f"Prompt: {sample['prompt']}\nChosen: {sample['chosen'][:200]}...\nRejected: {sample['rejected'][:200]}..."
        return None
    
    def __len__(self):
        return len(self.conversations)


def setup_qwen_model(model_name="Qwen/Qwen2.5-7B-Instruct"):
    """Setup Qwen2.5 model and tokenizer for DPO training."""
    
    # Configure for efficient training with 4-bit quantization
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    # )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2"  # Use flash attention if available
    )
    
    return model, tokenizer


class DPOTrainingPipeline:
    """Main DPO training pipeline class."""
    
    def __init__(self, jsonl_file, model_name="Qwen/Qwen2.5-7B-Instruct", output_dir="./qwen_dpo_model"):
        self.jsonl_file = jsonl_file
        self.model_name = model_name
        self.output_dir = output_dir
        self.dataset = None
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def prepare_data(self):
        """Load and prepare dataset."""
        print("=== Preparing Dataset ===")
        self.dataset = ConversationDataset(self.jsonl_file)
        
        # Show sample
        sample = self.dataset.get_sample()
        if sample:
            print("\nSample conversation format:")
            print("-" * 50)
            print(sample)
            print("-" * 50)
        
        return self.dataset.to_hf_dataset()
    
    def setup_model(self):
        """Setup model and tokenizer."""
        print("\n=== Setting up Model ===")
        self.model, self.tokenizer = setup_qwen_model(self.model_name)
        print(f"Model loaded: {self.model_name}")
    
    def configure_training(self):
        """Configure training arguments and LoRA for DPO."""
        
        # LoRA configuration optimized for Qwen2.5
        # peft_config = LoraConfig(
        #     lora_alpha=32,
        #     lora_dropout=0.05,
        #     r=16,
        #     bias="none",
        #     task_type="CAUSAL_LM",
        #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        # )
        
        # Training arguments for DPO
        training_args = DPOConfig(
            output_dir=self.output_dir,
            per_device_train_batch_size=1,  # Smaller batch size for DPO due to memory usage
            gradient_accumulation_steps=16,  # Increase accumulation to maintain effective batch size
            warmup_steps=100,
            num_train_epochs=1,
            learning_rate=1e-6,
            bf16=True,
            logging_steps=10,
            save_steps=500,
            save_strategy="steps",
            remove_unused_columns=False,
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
        )
        
        return training_args #, peft_config
    
    def train(self):
        """Execute the full DPO training pipeline."""
        print("Starting Qwen2.5 DPO Training Pipeline")
        
        # Prepare data
        hf_dataset = self.prepare_data()
        
        # Setup model
        self.setup_model()
        
        # Configure training
        training_args = self.configure_training()
        
        print(f"\n=== Training Configuration ===")
        print(f"Dataset size: {len(hf_dataset)}")
        print(f"Model: {self.model_name}")
        print(f"Output directory: {self.output_dir}")
        print(f"Epochs: {training_args.num_train_epochs}")
        print(f"Batch size: {training_args.per_device_train_batch_size}")
        print(f"Learning rate: {training_args.learning_rate}")
        
        # Initialize DPO trainer
        self.trainer = DPOTrainer(
            model=self.model,
            train_dataset=hf_dataset,
            # peft_config=peft_config,
            args=training_args,
        )
        
        print("\n=== Starting DPO Training ===")
        # Start training
        self.trainer.train()
        
        # Save model
        print("\n=== Saving Model ===")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Model saved to {self.output_dir}")


def main():
    """Main execution function."""
    
    # Configuration
    JSONL_FILE = "semi_automated_dataset_creation/processed_decomposed_dataset.jsonl"  # Replace with your file path
    MODEL_NAME = "/projects/p32722/Models/Qwen2.5-7B-Instruct/"
    OUTPUT_DIR = "/projects/p32722/Models/qwen2.5_dpo_model"
    
    # Initialize and run training pipeline
    pipeline = DPOTrainingPipeline(
        jsonl_file=JSONL_FILE,
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR
    )
    
    # Execute training
    pipeline.train()


if __name__ == "__main__":
    main()