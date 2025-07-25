import argparse
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
)

def load_model(model_name):
    # Try loading as CausalLM (decoder-only)
    try:
        return AutoModelForCausalLM.from_pretrained(model_name)
    except Exception:
        # Try loading as MaskedLM (encoder-only)
        try:
            return AutoModelForMaskedLM.from_pretrained(model_name)
        except Exception:
            # Try loading as Seq2Seq (encoder-decoder)
            try:
                return AutoModelForSeq2SeqLM.from_pretrained(model_name)
            except Exception:
                # Fallback to generic AutoModel
                return AutoModel.from_pretrained(model_name)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def format_param_count(n):
    if n >= 1_000_000_000:
        return f"{n:,} ({n/1_000_000_000:.2f} billion)"
    elif n >= 1_000_000:
        return f"{n:,} ({n/1_000_000:.2f} million)"
    else:
        return f"{n:,}"

def main():
    parser = argparse.ArgumentParser(description="Count parameters of a Hugging Face model.")
    parser.add_argument('--model_name', type=str, required=True, help='Model name from Hugging Face Hub (e.g., gpt2)')
    args = parser.parse_args()

    print(f"Loading model '{args.model_name}'...")
    model = load_model(args.model_name)
    total, trainable = count_parameters(model)
    print(f"Total parameters: {format_param_count(total)}")
    print(f"Trainable parameters: {format_param_count(trainable)}")

if __name__ == "__main__":
    main() 