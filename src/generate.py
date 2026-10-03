"""
src/generate.py
===============
Phase 53: CLI Interface for Text Generation using a trained MiniGPT model.
"""
import os
import sys
import argparse
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import MiniGPT
from src.dataset import GPTDataset
from src.tokenizer import CharTokenizer

def main():
    parser = argparse.ArgumentParser(description="Generate text using MiniGPT")
    parser.add_argument("--prompt", type=str, default="To be, or not to be", help="Starting text")
    parser.add_argument("--max_tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Creativity (higher = more random)")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k sampling cutoff")
    args = parser.parse_args()

    checkpoint_path = "checkpoints/minigpt_final.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: No checkpoint found at {checkpoint_path}. Run train.py first.")
        return

    # Load dataset to get the exact tokenizer mappings
    # (In a real system, you'd save the tokenizer vocab in the checkpoint)
    with open("data/input.txt", 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = CharTokenizer(text)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    config.vocab_size = checkpoint['vocab_size']
    
    # Initialize model and load weights
    model = MiniGPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Encode prompt
    context = tokenizer.encode(args.prompt)
    x = torch.tensor(context, dtype=torch.long).unsqueeze(0) # (1, len)

    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)
    
    # Generate
    y = model.generate(x, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k)
    
    # Decode and print
    generated_text = tokenizer.decode(y[0].tolist())
    print(generated_text)
    print("-" * 50)

if __name__ == "__main__":
    main()
