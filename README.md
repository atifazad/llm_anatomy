# LLM Inspector

This project provides a suite of utilities to inspect and analyze Hugging Face models programmatically, including parameter counting, tokenization details, attention visualization, context limits, and model configuration summaries.

## Setup

1. **Install [uv](https://github.com/astral-sh/uv):**
   ```bash
   pip install uv
   ```

2. **Create and activate a virtual environment:**
   ```bash
   uv venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

## Usage

Run the script with one of the following subcommands:

### 1. Count parameters
```bash
python llm_inspect.py count --model_name gpt2
```
**Example output:**
```
Loading model 'gpt2'...
Total parameters: 124,439,808 (124.44 million)
Trainable parameters: 124,439,808 (124.44 million)
```

### 2. Describe tokenization
```bash
python llm_inspect.py describe-tokenization --model_name gpt2
```
**Example output:**
```
Tokenizer class: <class 'transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast'>
Vocab size: 50257
Example text: jdkshd jskksd aioias
Tokens: ['jdk', 'sh', 'd', ' j', 'sk', 'ks', 'd', ' ai', 'oi', 'as']
Token IDs: [9926, 326, 72, 220, 11043, 11043, 72, 924, 924, 393]
```

### 3. Show attention patterns (requires matplotlib)
```bash
python llm_inspect.py show-attention --model_name gpt2 --text "The quick brown fox jumps over the lazy dog."
```
**Example output:**
```
Number of layers: 12
Layer 1 attention shape: torch.Size([12, 9, 9])
Layer 2 attention shape: torch.Size([12, 9, 9])
...
Layer 12 attention shape: torch.Size([12, 9, 9])
```
If `matplotlib` is installed, a heatmap window will pop up for Layer 1, Head 1. If not:
```
matplotlib not installed; skipping attention heatmap plot.
```

### 4. Show context limit (max sequence length)
```bash
python llm_inspect.py context-limit --model_name gpt2
```
**Example output:**
```
Tokenizer model_max_length: 1024
Config n_positions: 1024
Config max_position_embeddings: 1024
```

### 5. Model summary (common configuration properties)
```bash
python llm_inspect.py model-summary --model_name gpt2
```
**Example output:**
```
Model name: gpt2
--- Tokenizer Info ---
Tokenizer class: <class 'transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast'>
Vocab size: 50257
Model max length: 1024
Special tokens: ['<|endoftext|>']
Special token IDs: [50256]

--- Model Config Info ---
Model type: gpt2
Architectures: ['GPT2LMHeadModel']
Num hidden layers: 12
Num attention heads: 12
Hidden size: 768
n_positions: 1024
max_position_embeddings: 1024
Is encoder-decoder: False
Is decoder: True
Dropout (hidden): 0.1
Dropout (attention): 0.1
Activation function: gelu_new
Initializer range: 0.02
Layer norm epsilon: 1e-05
```

Replace `gpt2` with any model name from the [Hugging Face Model Hub](https://huggingface.co/models). 