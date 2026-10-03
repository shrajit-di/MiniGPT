"""
experiments/neural_network.py
=============================
Phases 3 to 8: Neural Networks, Forward Pass, Activations, 
Cross-Entropy Loss, and Backpropagation (Pure NumPy).
"""
import numpy as np

np.random.seed(42)

# ---------------------------------------------------------------------------
# Math & Dimensions Reminder:
# X: (Batch, Input_Dim)
# W: (Input_Dim, Output_Dim)
# y: X @ W + b -> (Batch, Output_Dim)
# ---------------------------------------------------------------------------

def relu(x):
    """Phase 5: Activation Function (ReLU) - Adds non-linearity."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU for Backpropagation."""
    return (x > 0).astype(float)

def softmax(logits):
    """
    Phase 5: Softmax - Converts raw scores (logits) into probabilities.
    Subtracts max for numerical stability.
    """
    exps = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)

def cross_entropy_loss(probs, targets):
    """
    Phase 6: Cross Entropy Loss
    Measures how far off our probabilities are from the true one-hot targets.
    Formula: -mean( sum(targets * log(probs)) )
    """
    m = targets.shape[0]
    # Add small epsilon to prevent log(0)
    p = np.clip(probs, 1e-15, 1 - 1e-15)
    log_likelihood = -np.log(p[range(m), targets])
    return np.sum(log_likelihood) / m

# --- Mini Neural Network Training ---
print("=== Training a 2-Layer NN in Pure NumPy ===")

# Dataset: 4 examples, 3 features
X = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.2, 0.1, 0.4]])
# Targets: Classes 0, 1, 2, 0
Y = np.array([0, 1, 2, 0]) 

# Network dimensions
input_dim = 3
hidden_dim = 8
output_dim = 3  # 3 classes

# Phase 4: Initialization
W1 = np.random.randn(input_dim, hidden_dim) * 0.1
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * 0.1
b2 = np.zeros((1, output_dim))

learning_rate = 0.1

for step in range(50):
    # ==========================
    # Phase 4: FORWARD PASS
    # ==========================
    # Layer 1
    Z1 = X @ W1 + b1       # Linear projection: (4, 3) @ (3, 8) -> (4, 8)
    A1 = relu(Z1)          # Activation

    # Layer 2
    Z2 = A1 @ W2 + b2      # Logits: (4, 8) @ (8, 3) -> (4, 3)
    probs = softmax(Z2)    # Probabilities: (4, 3)

    # Loss
    loss = cross_entropy_loss(probs, Y)
    
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss:.4f}")

    # ==========================
    # Phase 8: BACKPROPAGATION (Chain Rule)
    # ==========================
    m = X.shape[0]
    
    # Gradient of CrossEntropy + Softmax is simply: probs - true_labels
    dZ2 = probs.copy()
    dZ2[range(m), Y] -= 1
    dZ2 /= m  # shape: (4, 3)

    # Gradients for Layer 2
    dW2 = A1.T @ dZ2       # (8, 4) @ (4, 3) -> (8, 3)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # Gradients for Layer 1
    dA1 = dZ2 @ W2.T       # (4, 3) @ (3, 8) -> (4, 8)
    dZ1 = dA1 * relu_derivative(Z1) # Chain rule through ReLU
    
    dW1 = X.T @ dZ1        # (3, 4) @ (4, 8) -> (3, 8)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # ==========================
    # Phase 7: GRADIENT DESCENT (Weight Update)
    # ==========================
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

print("Training complete. Final probabilities for first example:")
print(np.round(probs[0], 3))
