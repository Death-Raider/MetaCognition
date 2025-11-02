# HiPO-DPO: Direct Preference Optimization with Hierarchical Prompt Optimization

## Introduction

This repository contains the implementation for training language models using Direct Preference Optimization (DPO) with HiPO (Hierarchical Prompt Optimization) format. HiPO enhances model responses by structuring outputs into three components:

1. **Refined Query (Rq)**: A clarified and improved version of the user's original query
2. **Meta Thinking (Mt)**: The model's reasoning process and approach to solving the problem
3. **Answer (Ra)**: The final response to the query

By training models with DPO on HiPO-formatted data, we encourage the model to not only provide accurate answers but also demonstrate transparent reasoning and query understanding.

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 16GB+ GPU memory (for 7B models)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/hipo-dpo.git
cd hipo-dpo

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers datasets trl peft accelerate bitsandbytes
```

### Required Packages

```bash
pip install torch>=2.0.0
pip install transformers>=4.36.0
pip install datasets>=2.14.0
pip install trl>=0.7.0
pip install peft>=0.7.0
pip install accelerate>=0.25.0
pip install bitsandbytes>=0.41.0
```

## Dataset Format

Your JSONL dataset should contain entries with the following structure:

```json
{
  "query": "What is machine learning?",
  "output_a": "Complete response A...",
  "output_b": "Complete response B...",
  "Rq_a": "Refined query A",
  "Mt_a": "Meta thinking A",
  "Ra_a": "Answer A",
  "Rq_b": "Refined query B",
  "Mt_b": "Meta thinking B",
  "Ra_b": "Answer B"
}
```

Place your dataset at: `semi_automated_dataset_creation/processed_decomposed_dataset.jsonl`

## General Usage

### 1. Standard DPO Training (Using `output_a/b`)

For standard DPO training without HiPO format:

```python
# In main() function, set:
USE_HIPO_FORMAT = False

# Run training
python dpo_training.py
```

This mode uses the pre-formatted `output_a` and `output_b` fields directly.

### 2. HiPO-DPO Training (Using `Rq + Mt + Ra`)

For training with HiPO format:

```python
# In main() function, set:
USE_HIPO_FORMAT = True
HIPO_INSTRUCTION_PATH = "Stage_5_HiPO_1Pass/instructions/instruction_few_shot.txt"

# Run training
python dpo_training.py
```

This mode:
- Constructs outputs from `Rq_a`, `Mt_a`, `Ra_a` components
- Prepends HiPO instructions to each query
- Trains the model to generate structured responses

### 3. Configuration

Modify training parameters in the `main()` function:

```python
# Model configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"  # or "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "Stage_5_HiPO_1Pass/models_saved/MistralModel_dpo"
JSONL_FILE = "semi_automated_dataset_creation/processed_decomposed_dataset.jsonl"

# HiPO configuration
USE_HIPO_FORMAT = True  # Toggle between standard and HiPO mode
HIPO_INSTRUCTION_PATH = "Stage_5_HiPO_1Pass/instructions/instruction_few_shot.txt"
```

### 4. Training Hyperparameters

Adjust training settings in `configure_training()`:

```python
training_args = DPOConfig(
    output_dir=self.output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    warmup_steps=100,
    num_train_epochs=5,
    learning_rate=1e-6,
    bf16=True,
    logging_steps=10,
    save_steps=500,
)
```

### 5. Using Trained Models

After training, load and use your model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "Stage_5_HiPO_1Pass/models_saved/MistralModel_dpo"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# Generate response
prompt = "What is photosynthesis?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Project Structure

```
.
├── dpo_training.py                          # Main training script
├── Stage_5_HiPO_1Pass/
│   ├── instructions/
│   │   └── instruction_few_shot.txt         # HiPO instructions
│   └── models_saved/
│       └── MistralModel_dpo/                # Saved model checkpoint
├── semi_automated_dataset_creation/
│   └── processed_decomposed_dataset.jsonl   # Training dataset
└── README.md
```

## HiPO Instructions File

Create `Stage_5_HiPO_1Pass/instructions/instruction_few_shot.txt` with your HiPO prompt instructions:

```
You are an AI assistant that provides structured responses.

For each query, you should:
1. First, refine and clarify the user's query
2. Then, explain your reasoning and approach
3. Finally, provide the complete answer

Format your response as:
[Refined Query]
<your refined query>

[Meta Thinking]
<your reasoning process>

[Answer]
<your final answer>
```

## Training Tips

1. **Memory Management**: For large models, consider enabling quantization:
   ```python
   bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_compute_dtype=torch.bfloat16,
   )
   ```

2. **Batch Size**: Adjust based on your GPU memory:
   - 24GB VRAM: `batch_size=4, gradient_accumulation_steps=16`
   - 40GB VRAM: `batch_size=8, gradient_accumulation_steps=8`
   - 80GB VRAM: `batch_size=16, gradient_accumulation_steps=4`

3. **Dataset Size**: Start with a subset for testing:
   ```python
   # In prepare_data()
   dataset = dataset.select(range(1000))  # Use first 1000 examples
   ```

## General Results

*(Results to be added after experiments)*

Our preliminary experiments show:
- **Improved Response Quality**: Models trained with HiPO-DPO demonstrate better structured reasoning
- **Enhanced Transparency**: The meta-thinking component provides insight into the model's decision-making
- **Query Understanding**: Refined queries indicate improved comprehension of user intent

Detailed results including benchmark scores, human evaluations, and ablation studies will be provided in the full paper.

## Troubleshooting

### Common Issues

**Out of Memory Error**
```bash
# Reduce batch size or enable gradient checkpointing
gradient_checkpointing=True
per_device_train_batch_size=2
```

**Tokenizer Warnings**
```python
# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

**Missing HiPO Instructions File**
```bash
# Create the directory and file
mkdir -p Stage_5_HiPO_1Pass/instructions
touch Stage_5_HiPO_1Pass/instructions/instruction_few_shot.txt
# Add your instructions to the file
```

## Citation

*(To be added upon publication)*

If you use this code or method in your research, please cite:

```bibtex
@article{yourname2024hipo,
  title={HiPO-DPO: Hierarchical Prompt Optimization with Direct Preference Optimization},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## Paper

[Link to paper will be added here]

## License

[Your chosen license, e.g., MIT, Apache 2.0]

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: your.email@example.com

## Acknowledgments

This work builds upon:
- [TRL](https://github.com/huggingface/trl) for DPO implementation
- [Transformers](https://github.com/huggingface/transformers) by Hugging Face
- [PEFT](https://github.com/huggingface/peft) for parameter-efficient fine-tuning
