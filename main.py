# ── CLI smoke test ───────────────────────────────────────────────────────────────────

import gpt
import torch


if __name__ == "__main__":
    config = gpt.ModelConfig()
    print(f"\nModel configuration:\n{config}\n")

    model = gpt.TinyStoriesModel(config)

    batch_size = 2
    input_ids = torch.randint(0, config.vocab_size, (batch_size, config.seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, config.seq_len))

    model.eval()
    logits, loss = model(input_ids, labels)
    print(f"Input shape:  {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss:         {loss}")

    prompt = input_ids[:1, :10]
    generated = model.generate(prompt, max_new_tokens=20, temperature=1.0)
    print(f"\nGeneration: {prompt.shape} → {generated.shape}")
    print("\nSmoke test passed!")

def main():
    print("Hello from tinystories!")


if __name__ == "__main__":
    main()
