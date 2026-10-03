"""
experiments/python_foundations.py
=================================
Core Python mechanics essential for building Transformer models:
1. Reference semantics & in-place mutations (avoiding silent bugs in tensors)
2. Slicing for sequence batching (context windowing)
3. Vocabulary mapping via dictionary comprehensions
4. Callable objects (__call__) mimicking PyTorch nn.Module
"""

from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# 1. Callable Module Pattern (Identical to how PyTorch nn.Module works)
# ---------------------------------------------------------------------------
class TinyLinear:
    """
    Simulates a 1D linear layer: y = x * weight + bias
    In Java: public class TinyLinear { ... public double compute(double x) { ... } }
    In Python: We define __call__ so instance(x) runs the forward pass!
    """
    def __init__(self, weight: float, bias: float) -> None:
        self.weight = weight
        self.bias = bias

    def forward(self, x: float) -> float:
        return x * self.weight + self.bias

    def __call__(self, x: float) -> float:
        # In PyTorch, __call__ handles forward hooks, profilers, and autograd graph registration
        return self.forward(x)


# ---------------------------------------------------------------------------
# 2. Tokenizer Mappings (Dict Comprehensions)
# ---------------------------------------------------------------------------
def build_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Extracts unique characters from text and builds bi-directional lookup tables.
    """
    unique_chars = sorted(list(set(text)))
    char_to_id = {ch: idx for idx, ch in enumerate(unique_chars)}
    id_to_char = {idx: ch for idx, ch in enumerate(unique_chars)}
    return char_to_id, id_to_char


# ---------------------------------------------------------------------------
# 3. Context Window Slicing (Autoregressive Next-Token Pairs)
# ---------------------------------------------------------------------------
def create_input_target_pairs(token_ids: List[int], block_size: int) -> Tuple[List[int], List[int]]:
    """
    Extracts input sequence and target sequence shifted by 1 position.
    Example:
      Tokens: [10, 20, 30, 40, 50] with block_size=3
      x (input)  = [10, 20, 30]
      y (target) = [20, 30, 40]
    """
    x = token_ids[:block_size]
    y = token_ids[1 : block_size + 1]
    return x, y


# ---------------------------------------------------------------------------
# Execution and Verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== 1. Testing Callable Layer ===")
    layer = TinyLinear(weight=2.5, bias=1.0)
    # Calling object as if it were a function:
    output = layer(4.0)  # 4.0 * 2.5 + 1.0 = 11.0
    print(f"Input: 4.0 -> Output: {output}")

    print("\n=== 2. Testing Vocabulary Mapping ===")
    sample_text = "hello world"
    char2id, id2char = build_vocab(sample_text)
    print(f"Unique characters ({len(char2id)}): {list(char2id.keys())}")
    encoded = [char2id[c] for c in "held"]
    print(f"Encoded 'held': {encoded}")
    decoded = "".join([id2char[i] for i in encoded])
    print(f"Decoded back: '{decoded}'")

    print("\n=== 3. Testing Context Slicing ===")
    full_sequence = [101, 102, 103, 104, 105, 106]
    block_size = 4
    x, y = create_input_target_pairs(full_sequence, block_size)
    print(f"Context (x): {x}")
    print(f"Target  (y): {y}")
    for t in range(len(x)):
        print(f"  When input is {x[:t+1]} -> Target to predict is: {y[t]}")
