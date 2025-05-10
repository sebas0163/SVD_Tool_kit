# SVD_Tool_Kit

**SVD_Tool_Kit** is a computational library in Python designed to compute different types of Singular Value Decompositions (SVDs), with the aim of optimizing the processing of multidimensional data and enhancing image analysis through advanced algorithmic implementations.

---

## 📦 Installation Requirements

This library requires the following dependencies:

numpy==2.2.5
scipy==1.15.3
numpy-quaternion

To install `numpy-quaternion`, use the following command:

```bash
python -m pip install --upgrade --force-reinstall numpy-quaternion

You can also install all dependencies at once using:

pip install -r requirements.txt

## ⚙️ Provided Decompositions

### `SVD`

Singular Value Decomposition (SVD) is a fundamental matrix factorization in linear algebra and data analysis. It provides a way to understand and manipulate matrices by decomposing them into simpler and more meaningful components.

### `Compact SVD`

A faster version of the standard SVD, commonly known as the "fast SVD" algorithm.

### `Generalized SVD (GSVD)`

Used to analyze two matrices \( A \in \mathbb{R}^{m_1 \times n} \) and \( B \in \mathbb{R}^{m_2 \times n} \), which share the same number of columns but not necessarily the same number of rows.

### `High Order SVD`

An extension of the GSVD that enables comparison of multiple large matrices with different row dimensions.

### `Joint SVD`

Extends SVD to multiple matrices that share a common structure. The goal is to find a shared basis that allows coherent representation of related matrices.

### `Tensor SVD`

An advanced technique for analyzing multidimensional data. It is based on the tensor-tensor product and decomposes a tensor into low-rank and sparse components.

### `Quaternion SVD`

A high-complexity algorithm for color image processing. It is based on the decomposition of quaternion matrices, which can efficiently represent color images.

---

## 🔧 Function Reference

### `svd(matrix, complete)`

- **Parameters**:
  - `matrix`: 2D array or matrix to decompose.
  - `complete`: Boolean flag; `True` for full SVD, `False` for compact SVD.
- **Returns**: `u`, `s`, `v` — the matrices from the decomposition \( A = U \Sigma V^T \)

---

### `GSVD(matrix_A, matrix_B)`

- **Parameters**:
  - `matrix_A`, `matrix_B`: Two matrices with the same number of columns and at least as many rows as columns.
- **Returns**: `u_1`, `u_2`, `d_a`, `d_b`, `x` — orthogonal bases for A and B, their diagonal matrices, and the shared transformation matrix `x`.

---

### `high_order_svd(matrix_list)`

- **Input**: A set of matrices with the same number of columns but potentially different row counts.
- **Returns**:
  - `matrix_u_list`: List of left singular vectors.
  - `sigma_list`: List of singular values.
  - `matrix_v`: Shared right singular vectors.

---

### `join_SVD(matrix_set)`

- **Input**: A set of matrices with different row and column dimensions.
- **Returns**:
  - `U`, `Dk`, `V`: Resulting matrices from the decomposition.
  - Error value, iteration count, and the final value calculated during the optimization.

---

### `Q_SVD(mat_Q)`

- **Parameter**: `mat_Q`, a quaternion matrix.
- **Returns**:
  - `mat_U_Q`, `mat_V_Q`: Quaternion orthogonal matrices.
  - `mat_S`: Singular value matrix.

---

### `T_SVD(tensor)`

- **Input**: A 3D tensor.
- **Returns**:
  - `tensor_U`, `tensor_S`, `tensor_V`: The SVD components of the tensor in the Fourier domain.

---

## ➕ Additional Utility Functions

- `t_product(tensor_a, tensor_b)`: Computes the t-product (tensor-tensor product).
- `transpose_tensor(tensor)`: Computes the transpose of a tensor.
- `frob_for_quat(quat_mat)`: Computes the Frobenius norm for a quaternion matrix.
- `dot_product_quat(mat_quat)`: Computes the dot product of a quaternion matrix.

---

## 🧪 Tests

There is a `test` folder in the project that includes test functions for every SVD implementation.

For example:

- `GSVD_test(m, n)` — Generates two matrices of dimensions m × n and runs performance and error analysis with time and accuracy plots.

Similar test functions exist for each implemented decomposition. These tests can be used to benchmark, visualize, and validate the different algorithms provided by the toolkit.
