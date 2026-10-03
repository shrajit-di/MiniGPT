"""
src/transformer.py
==================
Phases 20-23: Feed Forward Network, LayerNorm, and the Transformer Block.
"""
import torch
import torch.nn as nn
from attention import MultiHeadAttention

class FeedForward(nn.Module):
    """
    Phase 20: Feed Forward Network (MLP)
    Attention allows tokens to communicate. 
    The Feed Forward network allows tokens to individually "think" about what they just learned.
    """
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        # Standard GPT design: widen the representation by 4x inside the FFN
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),  # Gaussian Error Linear Unit (smoother than ReLU)
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    Phase 23: The full Transformer Block.
    Combines LayerNorm, Attention, and FeedForward with Residual Connections.
    """
    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        super().__init__()
        
        # Phase 22: Layer Normalization stabilizes deep neural network training
        self.ln_1 = nn.LayerNorm(n_embd)
        
        # The communication phase
        self.attn = MultiHeadAttention(n_embd, n_head, dropout, block_size)
        
        self.ln_2 = nn.LayerNorm(n_embd)
        
        # The computation phase
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        # Phase 21: Residual Connections (x = x + sublayer(x))
        # Notice we use the "Pre-Norm" architecture standard in modern GPTs:
        # We apply LayerNorm BEFORE the attention/FFwd block.
        
        # 1. Self-Attention (Communication)
        x = x + self.attn(self.ln_1(x))
        
        # 2. Feed-Forward (Computation)
        x = x + self.ffwd(self.ln_2(x))
        
        return x
