"""
src/tokenizer.py
================
Phase 10: Character-level Tokenizer.
Converts strings into lists of integer token IDs and vice versa.
"""
from typing import List, Dict

class CharTokenizer:
    def __init__(self, text_corpus: str):
        """
        Builds the vocabulary from the unique characters in the dataset.
        """
        # Get all unique characters, sort them for determinism
        chars = sorted(list(set(text_corpus)))
        self.vocab_size = len(chars)
        
        # Build bi-directional mappings
        self.char_to_id: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}
        
        print(f"Tokenizer initialized. Vocabulary size: {self.vocab_size}")

    def encode(self, text: str) -> List[int]:
        """Converts a string into a list of integer token IDs."""
        # If a character isn't in vocab, we simply skip or handle it. 
        # For this controlled MiniGPT, we assume all text matches the vocab.
        return [self.char_to_id[c] for c in text if c in self.char_to_id]

    def decode(self, token_ids: List[int]) -> str:
        """Converts a list of integer token IDs back into a string."""
        return "".join([self.id_to_char[i] for i in token_ids])

# Simple testing if run directly
if __name__ == "__main__":
    sample = "hello world"
    tok = CharTokenizer(sample)
    encoded = tok.encode("hello")
    decoded = tok.decode(encoded)
    print(f"Original: 'hello' -> Encoded: {encoded} -> Decoded: '{decoded}'")
