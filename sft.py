import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer
from peft import LoraConfig


class ConversationDataset:
    """Dataset class for loading and processing JSONL conversation data."""
    
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
        """Process a single JSONL entry and extract conversation pairs."""
        query = data.get('query', '').strip()
        
        if not query:
            return

        # Process Rq_a + Mt_a + Ra_a together for SFT
        output = data['Rq_a'] + ' ' + data['Mt_a'] + ' ' + data['Ra_a']
        conversation = self._format_conversation(query, output)
        self.conversations.append({"text": conversation})

    def _format_conversation(self, query, output):
        """Format conversation using Qwen2.5 chat template."""
        return f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    
    def to_hf_dataset(self):
        """Convert to HuggingFace Dataset."""
        return Dataset.from_list(self.conversations)
    
    def get_sample(self, index=0):
        """Get a sample conversation for inspection."""
        if self.conversations:
            return self.conversations[index]["text"]
        return None
    
    def __len__(self):
        return len(self.conversations)


def setup_qwen_model(model_name="Qwen/Qwen2.5-7B-Instruct"):
    """Setup Qwen2.5 model and tokenizer for training."""
    
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


class SFTTrainingPipeline:
    """Main training pipeline class."""
    
    def __init__(self, jsonl_file, model_name="Qwen/Qwen2.5-7B-Instruct", output_dir="./qwen_sft_model"):
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
            print(sample[:500] + "..." if len(sample) > 500 else sample)
            print("-" * 50)
        
        return self.dataset.to_hf_dataset()
    
    def setup_model(self):
        """Setup model and tokenizer."""
        print("\n=== Setting up Model ===")
        self.model, self.tokenizer = setup_qwen_model(self.model_name)
        print(f"Model loaded: {self.model_name}")
    
    def configure_training(self):
        """Configure training arguments and LoRA."""
        
        # LoRA configuration optimized for Qwen2.5
        # peft_config = LoraConfig(
        #     lora_alpha=32,
        #     lora_dropout=0.05,
        #     r=16,
        #     bias="none",
        #     task_type="CAUSAL_LM",
        #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        # )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=16,
            warmup_steps=100,
            num_train_epochs=5,
            learning_rate=5e-5,
            bf16=True,
            logging_steps=10,
            save_steps=500,
            save_strategy="steps",
            remove_unused_columns=False,
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            save_only_model=True
        )
        
        return training_args #peft_config
    
    def train(self):
        """Execute the full training pipeline."""
        print("Starting Qwen2.5 SFT Training Pipeline")
        
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
        
        # Initialize trainer
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=hf_dataset,
            # peft_config=peft_config,
            args=training_args,
        )
        
        print("\n=== Starting Training ===")
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
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
    OUTPUT_DIR = "Stage_5_HiPO_1Pass/models_saved/MistralModel_sft"
    
    # Initialize and run training pipeline
    pipeline = SFTTrainingPipeline(
        jsonl_file=JSONL_FILE,
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR
    )
    
    # Execute training
    pipeline.train()


if __name__ == "__main__":
    main()