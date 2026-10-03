"""
src/attention.py
================
Phase 19: Multi-Head Causal Self-Attention.
This is the core communication mechanism of MiniGPT.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by number of heads"
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_size = n_embd // n_head
        self.dropout = dropout

        # Instead of 3 separate matrices for Q, K, V, it's faster to do one large projection 
        # and slice it into 3 parts. 
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        
        # Output projection back to n_embd
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        # Dropout for regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # The causal mask (lower triangular). We register it as a buffer so PyTorch 
        # saves it in the state_dict but doesn't treat it as a learnable parameter.
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x):
        # B = Batch size, T = Sequence length (Time), C = Embedding dimension (Channels)
        B, T, C = x.size() 

        # 1. Compute Q, K, V
        # x is (B, T, C). c_attn(x) is (B, T, 3*C).
        qkv = self.c_attn(x)
        
        # Split the combined tensor into Q, K, V components
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # 2. Reshape for Multi-Head
        # We split the embedding dimension C into (n_head, head_size)
        # And we transpose so the Head dimension comes before the Time dimension
        # (B, T, C) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        # (B, n_head, T, hs) @ (B, n_head, hs, T) -> (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_size ** 0.5))
        
        # Apply the causal mask: Tokens can only look at past tokens (where mask == 1)
        # Everything in the future (where mask == 0) becomes -infinity.
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax converts scores to probabilities
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 4. Multiply by Values
        # (B, n_head, T, T) @ (B, n_head, T, hs) -> (B, n_head, T, hs)
        y = att @ v 
        
        # 5. Re-assemble all heads
        # Transpose back: (B, T, n_head, hs). 
        # .contiguous() forces the tensor to be contiguous in memory before reshaping
        # .view() flattens the heads back into C
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 6. Final linear projection
        out = self.resid_dropout(self.c_proj(y))
        
        return out
