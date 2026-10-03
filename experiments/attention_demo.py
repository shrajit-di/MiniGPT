"""
experiments/attention_demo.py
=============================
Phases 15-18: Step-by-Step Causal Self-Attention in PyTorch.
This demonstrates the mathematical core of the Transformer.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(42)

# Sequence parameters
B = 1       # Batch size
T = 4       # Sequence length (time steps)
C = 16      # Embedding dimension (channels)

print("=== 1. The Input ===")
# Imagine these are 4 tokens that have already been embedded
X = torch.randn(B, T, C)
print(f"Input X shape: {X.shape} (Batch, Time, Channels)")

# We create weights to project X into Q, K, and V
# In PyTorch, we typically use nn.Linear for this.
Wq = torch.randn(C, C)
Wk = torch.randn(C, C)
Wv = torch.randn(C, C)

print("\n=== 2. Q, K, V Projections ===")
# X @ W -> (1, 4, 16) @ (16, 16) -> (1, 4, 16)
Q = X @ Wq  # What each token is looking for
K = X @ Wk  # What each token contains
V = X @ Wv  # What each token will communicate if chosen
print(f"Q, K, V shapes: {Q.shape}")

print("\n=== 3. Attention Scores (Q dot K) ===")
# We want every token's Query to dot-product with every token's Key.
# Q is (B, T, C). K transpose needs to be (B, C, T)
# Q @ K^T -> (B, T, C) @ (B, C, T) -> (B, T, T)
scores = Q @ K.transpose(-2, -1) 
print(f"Raw scores shape: {scores.shape} (Batch, Time_Q, Time_K)")

print("\n=== 4. Scaling ===")
# Divide by sqrt(d_k) to prevent Softmax from getting too peaked
# (which would cause vanishing gradients)
d_k = C
scores = scores / (d_k ** 0.5)

print("\n=== 5. Causal Masking (No looking into the future!) ===")
# We create a lower-triangular matrix of ones
tril = torch.tril(torch.ones(T, T))
print("Lower Triangular Mask:\n", tril)

# Where the mask is 0, we fill the score with -infinity.
# When Softmax sees -infinity, it converts it to 0%.
scores = scores.masked_fill(tril == 0, float('-inf'))
print("\nMasked Scores:\n", torch.round(scores[0] * 10) / 10) # Rounded for readability

print("\n=== 6. Softmax (Attention Weights) ===")
# Convert scores to percentages that sum to 1 in each row
attention_weights = F.softmax(scores, dim=-1)
print("Attention Weights:\n", torch.round(attention_weights[0] * 100) / 100)
# Notice the first row is [1, 0, 0, 0] (Token 1 can only look at itself)
# The second row distributes attention over [Token 1, Token 2, 0, 0]

print("\n=== 7. Output ===")
# Finally, multiply the attention weights by the Values
# (B, T, T) @ (B, T, C) -> (B, T, C)
out = attention_weights @ V
print(f"Final output shape: {out.shape} -> Exact same shape as Input X!")
