import numpy as np
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
    if m >= n:
        M1 = A.T @ A
        D, V1 = np.linalg.eigh(M1) 
        order = np.argsort(D)[::-1] 
        D, V = D[order], V1[:, order]
        s = np.sqrt(np.maximum(D, 0))  
        #S = np.vstack([np.diag(s), np.zeros((m - n, n))])
        mask = s > np.finfo(float).eps 
        U1 = (A @ V[:, mask]) / s[mask]
        U2, _ = np.linalg.qr(np.random.rand(m, m - mask.sum()))
        U = np.hstack([U1, U2])
    else:
        M1 = A @ A.T
        D, U1 = np.linalg.eigh(M1)  
        order = np.argsort(D)[::-1]
        D, U = D[order], U1[:, order]
        s = np.sqrt(np.maximum(D, 0))
        #s = np.hstack([np.diag(s), np.zeros((m, n - m))])
        mask = s > np.finfo(float).eps
        V1 = ((A.T @ U[:, mask]) / s[mask])
        V2, _ = np.linalg.qr(np.random.rand(n, n - mask.sum()))
        V = np.hstack([V1, V2])
    return U, s, V