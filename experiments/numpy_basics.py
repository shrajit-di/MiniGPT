"""
experiments/numpy_basics.py
===========================
Foundational NumPy operations required for MiniGPT.
These concepts map 1:1 to PyTorch tensors used later.
"""

import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Tensors, Shapes, and Dtypes
# ---------------------------------------------------------------------------
print("=== 1. Tensors & Shapes ===")
# 0D (Scalar)
scalar = np.array(5.0)
print(f"Scalar: shape={scalar.shape}, ndim={scalar.ndim}")

# 1D (Vector)
vector = np.array([1, 2, 3, 4], dtype=np.float32)
print(f"Vector: shape={vector.shape}, ndim={vector.ndim}, type={vector.dtype}")

# 2D (Matrix) - Think of this as [Batch, Sequence]
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Matrix: shape={matrix.shape}, ndim={matrix.ndim}")

# ---------------------------------------------------------------------------
# 2. Reshaping and Transposing
# ---------------------------------------------------------------------------
print("\n=== 2. Reshape & Transpose ===")
# Original matrix is (2, 3)
# Reshape to (3, 2)
reshaped = matrix.reshape(3, 2)
print(f"Reshaped to (3, 2):\n{reshaped}")

# Transpose flips axes (2, 3) -> (3, 2). Mathematically different from reshape!
transposed = matrix.T
print(f"Transposed to (3, 2):\n{transposed}")

# ---------------------------------------------------------------------------
# 3. Broadcasting (The Magic of ML arrays)
# ---------------------------------------------------------------------------
print("\n=== 3. Broadcasting ===")
# Matrix is (2, 3). Let's add a vector of shape (3,).
# NumPy "broadcasts" the vector to each row automatically.
vec = np.array([10, 20, 30])
result = matrix + vec
print(f"Matrix (2,3) + Vector (3,) = \n{result}")

# ---------------------------------------------------------------------------
# 4. Matrix Multiplication (@) vs Element-wise (*)
# ---------------------------------------------------------------------------
print("\n=== 4. Matrix Multiplication ===")
A = np.array([[1, 2], [3, 4]])  # shape (2, 2)
B = np.array([[5, 6], [7, 8]])  # shape (2, 2)

# Element-wise multiplication
print(f"A * B (Element-wise):\n{A * B}")

# Matrix multiplication (Dot product of rows & columns)
# C[i, j] = Row i of A dot Col j of B
C = A @ B
print(f"A @ B (Matrix Multiplied):\n{C}")

# ---------------------------------------------------------------------------
# 5. Statistical Operations along Axes
# ---------------------------------------------------------------------------
print("\n=== 5. Aggregation & Axes ===")
data = np.array([[10, 20, 30], 
                 [100, 200, 300]]) # shape (2, 3)

# Mean over the entire array
print(f"Global Mean: {data.mean()}")

# Mean across rows (axis=0): collapses dimension 0 -> output shape (3,)
print(f"Mean across axis=0: {data.mean(axis=0)}")

# Variance & Standard Deviation (Crucial for LayerNorm)
variance = data.var(axis=1) # collapses dimension 1 -> output shape (2,)
std_dev = data.std(axis=1)
print(f"Variance (axis=1): {variance}")
print(f"Std Dev  (axis=1): {std_dev}")
