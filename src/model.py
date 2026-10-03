"""
src/model.py
============
Phases 24-26, 31-32: The complete MiniGPT Architecture and Generation logic.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformer import TransformerBlock

class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Phase 13: Token Embedding matrix (Vocab Size -> Embedding Dim)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Phase 14: Positional Embedding matrix (Sequence Length -> Embedding Dim)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        
        # Phase 24: Stacking Transformer Blocks sequentially
        self.blocks = nn.Sequential(*[
            TransformerBlock(config.n_embd, config.n_head, config.dropout, config.block_size) 
            for _ in range(config.n_layer)
        ])
        
        # Final LayerNorm before the classifier head
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Language Modeling Head: projects embeddings back to vocabulary size to get Logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        """
        idx: (Batch, Time) tensor of integers
        targets: (Batch, Time) tensor of integers (optional)
        """
        B, T = idx.size()
        
        # 1. Embeddings
        tok_emb = self.token_embedding(idx) # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        
        # 2. Transformer Blocks
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        
        # 3. Logits (Phase 26)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # PyTorch's cross_entropy expects a 2D tensor for logits and 1D for targets
            B, T, C = logits.shape
            logits_view = logits.view(B * T, C)
            targets_view = targets.view(B * T)
            # Calculates loss directly from logits for numerical stability! (Avoids explicit Softmax)
            loss = F.cross_entropy(logits_view, targets_view)
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Phase 31 & 32: Autoregressive Text Generation with Temperature and Top-k
        """
        for _ in range(max_new_tokens):
            # If sequence exceeds block_size, crop it to the maximum context window
            idx_cond = idx[:, -self.config.block_size:]
            
            # Get predictions for the sequence
            logits, _ = self(idx_cond)
            
            # We only care about the VERY LAST token's prediction
            logits = logits[:, -1, :] # (B, vocab_size)
            
            # Apply Temperature scaling
            # T > 1.0 makes distribution flatter (more random)
            # T < 1.0 makes distribution sharper (more greedy/confident)
            if temperature != 1.0:
                logits = logits / temperature
                
            # Apply Top-k sampling: Keep only the top k probabilities, zero out the rest
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Mask out anything smaller than the smallest value in the top-k
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # Append sampled token to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx
