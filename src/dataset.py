"""
src/dataset.py
==============
Phases 11-12: Dataset Creation, Train/Val Split, and Batch Generation.
"""
import os
import torch
from tokenizer import CharTokenizer

class GPTDataset:
    def __init__(self, file_path: str, block_size: int, split_ratio: float = 0.9):
        self.block_size = block_size
        
        # 1. Load data
        if not os.path.exists(file_path):
            self._create_dummy_data(file_path)
            
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # 2. Tokenize the entire dataset
        self.tokenizer = CharTokenizer(text)
        self.vocab_size = self.tokenizer.vocab_size
        
        # Convert all text to one massive 1D tensor of integers
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        
        # 3. Train/Validation Split
        n = int(split_ratio * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        
        print(f"Dataset loaded. Train tokens: {len(self.train_data)}, Val tokens: {len(self.val_data)}")

    def _create_dummy_data(self, file_path: str):
        print(f"Creating sample dataset at {file_path}")
        sample_text = (
            "To be, or not to be, that is the question: "
            "Whether 'tis nobler in the mind to suffer "
            "The slings and arrows of outrageous fortune, "
            "Or to take arms against a sea of troubles "
            "And by opposing end them. To die—to sleep, "
            "No more; and by a sleep to say we end "
            "The heart-ache and the thousand natural shocks "
            "That flesh is heir to: 'tis a consummation "
            "Devoutly to be wish'd."
        )
        # Duplicate to make it slightly larger for batching
        sample_text = (sample_text + "\n") * 100 
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(sample_text)

    def get_batch(self, split: str, batch_size: int):
        """
        Randomly samples a batch of context windows (x) and targets (y).
        """
        data = self.train_data if split == 'train' else self.val_data
        
        # Generate `batch_size` random starting indices
        # We ensure we don't go out of bounds by subtracting block_size
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        
        # Stack the slices into a single 2D tensor (Batch, Block_Size)
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        
        return x, y

# Simple testing
if __name__ == "__main__":
    ds = GPTDataset("data/input.txt", block_size=8)
    xb, yb = ds.get_batch('train', batch_size=4)
    print("\nBatch Input (x) shape:", xb.shape) # Expected: (4, 8)
    print("Batch Target (y) shape:", yb.shape) # Expected: (4, 8)
    
    print("\nFirst sequence in batch (x):", ds.tokenizer.decode(xb[0].tolist()))
    print("First sequence in batch (y):", ds.tokenizer.decode(yb[0].tolist()))
