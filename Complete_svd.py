import numpy as np
import time as tm
"""
    The function `full_svd` computes the full singular value decomposition (SVD) of a given matrix `A`.
    
    :param A: The function `full_svd` you provided is used to compute the full Singular Value
    Decomposition (SVD) of a matrix `A`. The parameters of the function are as follows:
    :return: The function `full_svd` returns the matrices U, S, and V which represent the full singular
    value decomposition of the input matrix A. U is the left singular vectors matrix, S is the singular
    values matrix, and V is the right singular vectors matrix.
    """
def full_svd(A):
    m, n = A.shape
    if m > n:
        M1 = A.T @ A 
        D, V1 = np.linalg.eig(M1) #Takes de eigenvectos and eigenvalues of the matrix
        y1 = D # y1 is the eigenvects
        const = n * np.max(y1) * np.finfo(float).eps
        y2 = y1 > const
        rA = np.sum(y2) #matrix rank
        y3 = y1 * y2
        s1 = np.sort(np.sqrt(y3))[::-1]
        order = np.argsort(np.sqrt(y3))[::-1]
        V = V1[:, order]
        S = np.vstack([np.diag(s1), np.zeros((m - len(s1), n))])
        U1 = (A @ V[:, :rA]) / s1[:rA]
        U2, _ = np.linalg.qr(np.hstack([U1, np.random.rand(m, m - rA)]))
        U = np.hstack([U1, U2[:, rA:m]])
    else:
        M1 = A @ A.T
        D, U1 = np.linalg.eig(M1)
        y1 = D
        const = m * np.max(y1) * np.finfo(float).eps
        y2 = y1 > const
        rA = np.sum(y2)  # rango de la matriz
        y3 = y1 * y2
        s1 = np.sort(np.sqrt(y3))[::-1]
        order = np.argsort(np.sqrt(y3))[::-1]
        U = U1[:, order]
        S = np.hstack([np.diag(s1), np.zeros((m, n - len(s1)))])
        V1 = (A.T @ U[:, :rA]) / s1[:rA]
        V2, _ = np.linalg.qr(np.hstack([V1, np.random.rand(n, n - rA)]))
        V = np.hstack([V1, V2[:, rA:n]])
    return U,S,V