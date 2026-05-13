"""Interactive text generation from a trained checkpoint.

Usage:
    uv run python generate.py ckpt/bpe_16384-v3.1.1/latest.pt
    uv run python generate.py ckpt/bpe_16384-v3.1.1/latest.pt --temperature 0.5
    uv run python generate.py ckpt/bpe_16384-v3.1.1/latest.pt --max-tokens 200
"""

import argparse
from pathlib import Path

import torch

from gpt import ModelConfig, TinyStoriesModel
from gpt_teacher import TeacherConfig, TeacherModel
from train import _load_tokenizer, _strip_compile_prefix

DATA_DIR = Path("data")


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    if isinstance(config, TeacherConfig):
        model = TeacherModel(config).to(device)
    else:
        model = TinyStoriesModel(config).to(device)
    model.load_state_dict(_strip_compile_prefix(ckpt["model"]))
    model.eval()
    return model, config


def main():
    p = argparse.ArgumentParser(description="Interactive text generation")
    p.add_argument("checkpoint", help="Path to .pt checkpoint")
    p.add_argument("--tok-name", default=None,
                   help="Tokenizer name (auto-detected from run name if omitted)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=200)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Auto-detect tokenizer from checkpoint path
    if args.tok_name is None:
        run_name = Path(args.checkpoint).parent.name
        tok_name = run_name.rsplit("-v", 1)[0]
    else:
        tok_name = args.tok_name

    tok_path = DATA_DIR / tok_name / "tokenizer.json"
    if not tok_path.exists():
        print(f"Tokenizer not found: {tok_path}")
        return
    tokenizer = _load_tokenizer(tok_path)
    print(f"Tokenizer: {tok_name} (vocab_size={tokenizer.vocab_size})")

    model, config = load_model(args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Temperature: {args.temperature}, Max tokens: {args.max_tokens}\n")

    print("Enter a prompt (empty line = 'Once upon a time', Ctrl-C to quit):\n")
    while True:
        try:
            prompt = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not prompt:
            prompt = "Once upon a time"

        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=device)

        output = model.generate(input_tensor, max_new_tokens=args.max_tokens,
                                temperature=args.temperature)
        tokens = output[0].tolist()
        # Truncate at first EOS token
        eos = tokenizer.eos_id
        if eos in tokens[len(input_ids):]:
            tokens = tokens[:tokens.index(eos, len(input_ids))]
        text = tokenizer.decode(tokens)
        print(f"\n{text}\n")


if __name__ == "__main__":
    main()
