# LLM Parameter Counter

This project allows you to download any Hugging Face model and count its parameters programmatically.

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

Run the script to count parameters for a given Hugging Face model:

```bash
python count_parameters.py --model_name gpt2
```

Replace `gpt2` with any model name from the [Hugging Face Model Hub](https://huggingface.co/models). 