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

### 2. Describe tokenization
```bash
python llm_inspect.py describe-tokenization --model_name gpt2
```

### 3. Show attention patterns (requires matplotlib)
```bash
python llm_inspect.py show-attention --model_name gpt2 --text "The quick brown fox jumps over the lazy dog."
```

### 4. Show context limit (max sequence length)
```bash
python llm_inspect.py context-limit --model_name gpt2
```

### 5. Model summary (common configuration properties)
```bash
python llm_inspect.py model-summary --model_name gpt2
```

Replace `gpt2` with any model name from the [Hugging Face Model Hub](https://huggingface.co/models). 