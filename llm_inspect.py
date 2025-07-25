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

def describe_tokenization(model_name, text="jdkshd jskksd aioias"):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer class: {type(tokenizer)}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Example text: {text}")
    print(f"Tokens: {tokenizer.tokenize(text)}")
    print(f"Token IDs: {tokenizer.encode(text)}")

def show_attention(model_name, text):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    try:
        import matplotlib.pyplot as plt
        has_matplotlib = True
    except ImportError:
        has_matplotlib = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions
    print(f"Number of layers: {len(attentions)}")
    for i, attn in enumerate(attentions):
        print(f"Layer {i+1} attention shape: {attn.shape}")
    if has_matplotlib:
        attn = attentions[0][0, 0].detach().numpy()  # (seq_len, seq_len)
        plt.imshow(attn, cmap='viridis')
        plt.colorbar()
        plt.title("Layer 1, Head 1 Attention")
        plt.xlabel("Input Token Index")
        plt.ylabel("Output Token Index")
        plt.show()
    else:
        print("matplotlib not installed; skipping attention heatmap plot.")

def show_context_limit(model_name):
    from transformers import AutoTokenizer, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    print(f"Tokenizer model_max_length: {tokenizer.model_max_length}")
    # Try common config attributes
    if hasattr(config, 'n_positions'):
        print(f"Config n_positions: {config.n_positions}")
    if hasattr(config, 'max_position_embeddings'):
        print(f"Config max_position_embeddings: {config.max_position_embeddings}")

def model_summary(model_name):
    from transformers import AutoTokenizer, AutoConfig
    print(f"Model name: {model_name}")
    print("--- Tokenizer Info ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer class: {type(tokenizer)}")
    print(f"Vocab size: {getattr(tokenizer, 'vocab_size', 'N/A')}")
    print(f"Model max length: {getattr(tokenizer, 'model_max_length', 'N/A')}")
    print(f"Special tokens: {getattr(tokenizer, 'all_special_tokens', 'N/A')}")
    print(f"Special token IDs: {getattr(tokenizer, 'all_special_ids', 'N/A')}")
    print()
    print("--- Model Config Info ---")
    config = AutoConfig.from_pretrained(model_name)
    print(f"Model type: {getattr(config, 'model_type', 'N/A')}")
    print(f"Architectures: {getattr(config, 'architectures', 'N/A')}")
    print(f"Num hidden layers: {getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 'N/A'))}")
    print(f"Num attention heads: {getattr(config, 'num_attention_heads', getattr(config, 'n_head', 'N/A'))}")
    print(f"Hidden size: {getattr(config, 'hidden_size', getattr(config, 'n_embd', 'N/A'))}")
    if hasattr(config, 'n_positions'):
        print(f"n_positions: {config.n_positions}")
    if hasattr(config, 'max_position_embeddings'):
        print(f"max_position_embeddings: {config.max_position_embeddings}")
    print(f"Is encoder-decoder: {getattr(config, 'is_encoder_decoder', 'N/A')}")
    print(f"Is decoder: {getattr(config, 'is_decoder', 'N/A')}")
    print(f"Dropout (hidden): {getattr(config, 'hidden_dropout_prob', 'N/A')}")
    print(f"Dropout (attention): {getattr(config, 'attention_probs_dropout_prob', 'N/A')}")
    print(f"Activation function: {getattr(config, 'hidden_act', 'N/A')}")
    print(f"Initializer range: {getattr(config, 'initializer_range', 'N/A')}")
    print(f"Layer norm epsilon: {getattr(config, 'layer_norm_eps', 'N/A')}")

def main():
    parser = argparse.ArgumentParser(description="Utilities for Hugging Face models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: count
    count_parser = subparsers.add_parser("count", help="Count parameters of a Hugging Face model.")
    count_parser.add_argument('--model_name', type=str, required=True, help='Model name from Hugging Face Hub (e.g., gpt2)')

    # Subcommand: describe-tokenization
    desc_parser = subparsers.add_parser("describe-tokenization", help="Describe the tokenization for the given model.")
    desc_parser.add_argument('--model_name', type=str, required=True, help='Model name from Hugging Face Hub (e.g., gpt2)')

    # Subcommand: show-attention
    attn_parser = subparsers.add_parser("show-attention", help="Show attention patterns for a given model and text.")
    attn_parser.add_argument('--model_name', type=str, required=True, help='Model name from Hugging Face Hub (e.g., gpt2)')
    attn_parser.add_argument('--text', type=str, required=True, help='Input text to analyze attention patterns.')

    # Subcommand: context-limit
    ctx_parser = subparsers.add_parser("context-limit", help="Show the context window (max sequence length) for a model.")
    ctx_parser.add_argument('--model_name', type=str, required=True, help='Model name from Hugging Face Hub (e.g., gpt2)')

    # Subcommand: model-summary
    summary_parser = subparsers.add_parser("model-summary", help="Show a summary of common LLM configuration properties.")
    summary_parser.add_argument('--model_name', type=str, required=True, help='Model name from Hugging Face Hub (e.g., gpt2)')

    args = parser.parse_args()

    if args.command == "describe-tokenization":
        describe_tokenization(args.model_name)
    elif args.command == "count":
        print(f"Loading model '{args.model_name}'...")
        model = load_model(args.model_name)
        total, trainable = count_parameters(model)
        print(f"Total parameters: {format_param_count(total)}")
        print(f"Trainable parameters: {format_param_count(trainable)}")
    elif args.command == "show-attention":
        show_attention(args.model_name, args.text)
    elif args.command == "context-limit":
        show_context_limit(args.model_name)
    elif args.command == "model-summary":
        model_summary(args.model_name)

if __name__ == "__main__":
    main() 