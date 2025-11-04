## LoRA Fine-tuning

### Overview

This notebook demonstrates parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation) with SmolLM2-135M. LoRA trains only 1-2% of model parameters by adding small adapter matrices, achieving similar quality to full fine-tuning with significantly less compute and memory.

### What You'll Learn

- Understanding LoRA and how it differs from full fine-tuning
- Configuring LoRA hyperparameters (rank, alpha, target modules)
- Training adapters that are 100x smaller than full models
- Comparing LoRA performance with full fine-tuning
- When to use LoRA versus other methods

### Requirements

- Google Colab with T4 GPU (free tier)
- Approximately 10-15 minutes of training time
- No local setup required

### Model Information

Base Model: HuggingFaceTB/SmolLM2-135M-Instruct
- Total Parameters: 135 million
- Trainable Parameters with LoRA: ~1.6 million (1.2%)
- Adapter Size: 2-5 MB

### LoRA Configuration

```python
lora_r = 16              # LoRA rank
lora_alpha = 16          # Scaling factor
lora_dropout = 0.05      # Regularization
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",      # MLP
]
```

### Key Hyperparameters

Rank (r): Controls adapter capacity. Higher rank = more capacity but more parameters.
- r=8: Minimal, very fast
- r=16: Good default
- r=32: High quality
- r=64: Near full fine-tuning quality

Alpha: Scaling factor, typically set equal to rank.

Target Modules: Which layers receive adapters. More modules = higher capacity.

### Training Configuration

```python
batch_size = 2
gradient_accumulation_steps = 4
learning_rate = 2e-4  # Higher than full fine-tuning
max_steps = 50
```

### Output Files

```
./smollm2_lora_adapters/
├── adapter_config.json
├── adapter_model.bin      # Only 2-5 MB!
├── special_tokens_map.json
├── tokenizer_config.json
└── tokenizer.json

./smollm2_lora_merged/
└── [Full merged model ~540 MB]
```

### Performance Comparison

| Metric | Full Fine-tuning | LoRA |
|--------|------------------|------|
| Trainable Params | 135M (100%) | 1.6M (1.2%) |
| Training Time | ~18s/step | ~12s/step |
| Memory Usage | Higher | Lower |
| Adapter Size | 540 MB | 2-5 MB |
| Quality | Excellent | Very Good |

### Usage

#### Loading LoRA Adapters

```python
from unsloth import FastLanguageModel

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M-Instruct"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(model, "./smollm2_lora_adapters")

# Or load merged model directly
model, tokenizer = FastLanguageModel.from_pretrained(
    "./smollm2_lora_merged"
)
```

#### Swapping Adapters

```python
# Load base model once
base_model, tokenizer = FastLanguageModel.from_pretrained(base_model_name)

# Swap different task-specific adapters
model = PeftModel.from_pretrained(base_model, "adapter_task1")
# Generate for task 1

model = PeftModel.from_pretrained(base_model, "adapter_task2")
# Generate for task 2
```

### Advantages of LoRA

1. Memory Efficient: Trains 1-2% of parameters
2. Fast Training: 30% faster than full fine-tuning
3. Small Files: Adapters are only 2-5 MB, easy to share
4. Multiple Tasks: Swap adapters on same base model
5. Less Overfitting: Fewer parameters to overfit
6. Preserves Base Model: Original capabilities maintained

### When to Use LoRA

Recommended For:
- Limited GPU memory (under 24GB)
- Quick experimentation and iteration
- Multiple task-specific adaptations
- Small to medium datasets (100-10K samples)
- Sharing and distributing models
- Most production use cases

Not Recommended For:
- Huge datasets (100K+ samples) where full fine-tuning might excel
- Need absolute maximum performance
- Unlimited compute resources

### Troubleshooting

Issue: LoRA adapters not learning
- Increase rank (r=32 or 64)
- Increase learning rate
- Add more target modules
- Train for more steps

Issue: Quality worse than expected
- Compare with full fine-tuning baseline
- Check data quality
- Increase rank
- Tune alpha parameter

Issue: Outputs similar to base model
- Verify adapters are loaded
- Check training actually ran (loss decreased)
- Increase training steps
- Higher learning rate

### Extensions

#### Custom Target Modules

Target only query and value projections:
```python
target_modules = ["q_proj", "v_proj"]
```

Target all linear layers:
```python
target_modules = "all-linear"
```

#### Rank Experimentation

Try different ranks to find optimal trade-off:
```python
for r in [8, 16, 32, 64]:
    model = FastLanguageModel.get_peft_model(
        model, r=r, ...
    )
    train()
    evaluate()
```

#### Export to GGUF

```python
model.save_pretrained_gguf(
    "lora_model_gguf",
    tokenizer,
    quantization_method="q4_k_m"
)
```

### Resources

- LoRA Paper: https://arxiv.org/abs/2106.09685
- Unsloth LoRA Guide: https://docs.unsloth.ai



