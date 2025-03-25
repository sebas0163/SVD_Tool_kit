import numpy as np

def lu_factorization(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for i in range(n):
        L[i, i] = 1
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]
    
    return L, U

def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)
    
    for i in range(n):
        y[i] = b[i] - sum(L[i, j] * y[j] for j in range(i))
    
    return np.array(y)

def backward_substitution(U, y):
    n = len(y)
    x = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]
    
    return np.array(x)


"""
# Ejemplo de uso
A = np.array([[4, 3], [6, 3]], dtype=float)
b = np.array([10, 12], dtype=float)

L, U = lu_factorization(A)
y = forward_substitution(L, b)
x = backward_substitution(U, y)

print("Matriz L:")
print(L)
print("Matriz U:")
print(U)
print("Solución del sistema Ax = b:")
print(x)
"""
