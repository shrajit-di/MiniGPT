# MiniGPT — GPT-Style Language Model From Scratch

This project is a complete, educational implementation of a decoder-only Transformer language model built from scratch. It avoids all API shortcuts and external model weights, relying purely on mathematically explicit PyTorch tensor operations.

## Architecture

```text
                    ┌─────────────────┐
                    │   Input Text    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │    Tokenizer    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   Token IDs     │
                    └────────┬────────┘
                             ↓
             ┌───────────────┴───────────────┐
             ↓                               ↓
    Token Embedding                   Position Embedding
             │                               │
             └───────────────┬───────────────┘
                             ↓
                    ┌─────────────────┐
                    │ Transformer     │
                    │ Block 1         │
                    └────────┬────────┘
                             ↓
                            ...
                             ↓
                    ┌─────────────────┐
                    │ Transformer     │
                    │ Block N         │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   LayerNorm     │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │     LM Head     │
                    └────────┬────────┘
                             ↓
                         Logits
                             ↓
                     Next Token (Sampling)
```

## Tech Stack
- **Language**: Python 3.x
- **Numerical Math**: NumPy (for foundational experiments)
- **Deep Learning**: PyTorch (for actual model & autograd)
- **Visualization**: Matplotlib (loss curves)

## Features
- **Custom Tokenization**: Character-level encoding for educational simplicity.
- **Embeddings**: Both Token and Learned Positional embeddings.
- **Causal Self-Attention**: Hand-written scaled dot-product attention with lower-triangular masking.
- **Multi-Head Attention**: Parallelized communication heads.
- **Transformer Blocks**: Pre-norm architecture with GELU Feed-Forward Networks and Residual Connections.
- **Training Pipeline**: Cross-entropy loss computation without explicit softmax (numerical stability), optimized via AdamW.
- **Generation**: Autoregressive decoding with configurable Temperature and Top-k sampling.

## How it works
MiniGPT takes a sequence of characters, converts them to integers, embeds them into a dense vector space, and injects positional data. The Transformer blocks then allow these tokens to "communicate" with each past token via self-attention, while preventing them from seeing the future via a causal mask. The final layers project these learned representations back into the vocabulary space (Logits), allowing the model to probabilistically sample the most logical next character.

## Installation
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Training
To train the model on the dummy Shakespeare dataset:
```bash
python src/train.py
```
This will train the model, save `checkpoints/minigpt_final.pt`, and plot the loss to `outputs/loss_plot.png`.

## Generation
To generate text using your trained checkpoint:
```bash
python src/generate.py --prompt "To be, " --max_tokens 100 --temperature 0.8
```

## Limitations
This is an educational model.
- **Small Dataset**: Trained on a tiny text snippet.
- **Small Compute**: Parameter counts are in the thousands/millions, not billions.
- **Tokenization**: Character-level tokenization is highly inefficient compared to modern BPE (Byte-Pair Encoding). 
- It is NOT comparable to production LLMs, but mathematically acts identically.

## Future Improvements
- Implement Byte-Pair Encoding (BPE) tokenizer (e.g. `tiktoken`).
- Scale up the dataset (e.g. OpenWebText).
- Add Flash Attention for optimized GPU training.
- Implement weight-tying between embeddings and the LM head.
- Implement distributed training.
