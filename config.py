"""
config.py
=========
Central configuration for MiniGPT.
Adjust these values based on your hardware.
"""
import torch

class MiniGPTConfig:
    # --- Model Architecture ---
    block_size = 64      # Maximum context length (sequence length)
    n_embd = 128         # Embedding dimension
    n_head = 4           # Number of attention heads
    n_layer = 4          # Number of Transformer blocks
    dropout = 0.1        # Dropout rate for regularization
    
    # --- Training Parameters ---
    batch_size = 16
    learning_rate = 1e-3
    max_iters = 500      # Number of training steps
    eval_interval = 100  # How often to print loss
    
    # --- Device & System ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Will be populated dynamically based on tokenizer
    vocab_size = None 

config = MiniGPTConfig()
