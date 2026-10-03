"""
src/train.py
============
Phases 27-30: The Training Loop, Optimizer, and Loss Monitoring.
"""
import os
import sys
import torch
import matplotlib.pyplot as plt

# Add the parent directory to sys.path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from model import MiniGPT
from dataset import GPTDataset

def train():
    # 1. Load Dataset & Config
    print("Loading dataset...")
    dataset = GPTDataset("data/input.txt", block_size=config.block_size)
    config.vocab_size = dataset.vocab_size
    print(f"Device: {config.device}")

    # 2. Initialize Model
    model = MiniGPT(config).to(config.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    # 3. Setup Optimizer (Phase 28: AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    train_losses = []
    eval_steps = []
    
    # 4. Training Loop (Phase 27)
    model.train()
    for step in range(config.max_iters):
        # Sample a batch of data
        xb, yb = dataset.get_batch('train', config.batch_size)
        xb, yb = xb.to(config.device), yb.to(config.device)

        # Forward pass
        logits, loss = model(xb, yb)

        # Zero gradients, Backward pass, Optimizer step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Monitoring
        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            print(f"Step {step}/{config.max_iters} | Training Loss: {loss.item():.4f}")
            train_losses.append(loss.item())
            eval_steps.append(step)

    # 5. Checkpointing (Phase 33)
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/minigpt_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'vocab_size': config.vocab_size
    }, checkpoint_path)
    print(f"\nModel saved to {checkpoint_path}")

    # 6. Loss Visualization (Phase 36)
    os.makedirs("outputs", exist_ok=True)
    plt.plot(eval_steps, train_losses)
    plt.xlabel('Steps')
    plt.ylabel('Cross Entropy Loss')
    plt.title('MiniGPT Training Loss')
    plt.savefig("outputs/loss_plot.png")
    print("Loss plot saved to outputs/loss_plot.png")

if __name__ == "__main__":
    train()
