import numpy as np
"""
    The function `diag_rec` creates a new matrix with only the diagonal elements copied from the input
    matrix `A`.
    
    :param A: The function `diag_rec` takes a numpy array `A` as input and returns a new numpy array `X`
    with the diagonal elements of `A` copied over
    :return: The function `diag_rec` takes a square matrix `A` as input and creates a new matrix `X`
    with the same shape as `A`, where the diagonal elements of `X` are the same as the diagonal elements
    of `A` and all other elements are zero. The function then returns this new matrix `X`.
    """
def diag_rec(A):
    X = np.zeros_like(A)
    k = min(A.shape)
    for i in range(k):
        X[i, i] = A[i, i]
    return X
"""
    The function `join_SVD` performs Singular Value Decomposition (SVD) on a set of matrices iteratively
    to find the best approximation using matrix factorization.
    
    :param matrix_set: It seems like the description of the `matrix_set` parameter is missing. Could you
    please provide more information on what the `matrix_set` contains or represents? This will help in
    understanding the input data structure and how the function `join_SVD` is intended to operate on it
    :return: The function `join_SVD` returns the matrices U, Dk, V, the error value, the number of
    iterations performed, and the final value calculated.
    """
def join_SVD(matrix_set):
    K,m,n= matrix_set.shape
    AU = np.zeros((m, m))
    AV = np.zeros((n, n))
    value_of =0
    for k in range(K):
        Ak = matrix_set[k]
        AU += Ak @ Ak.T
        AV += Ak.T @ Ak
    _, U = np.linalg.eigh(AU)
    _, V = np.linalg.eigh(AV)
    iter_max = 10000
    tol = 1e-10
    Dk = np.zeros((K, m, n))
    for i in range(iter_max):
        for k in range(K): #DK act
            Dk[k] = diag_rec(U.T @ matrix_set[k] @ V)
        M = np.zeros((m, m))
        for k in range(K): #Update of the Matrix U
            M += matrix_set[k] @ V @ Dk[k].T
        Up, _, Vp = np.linalg.svd(M)
        U = Up @ Vp
        N = np.zeros((n, n))
        for k in range(K): #Update of the matrix V
            N += matrix_set[k].T @ U @ Dk[k]
        Uq, _, Vq = np.linalg.svd(N)
        V = Uq @ Vq
        value_of_n = 0
        for k in range(K):
            value_of_n += np.linalg.norm(matrix_set[k] - U @ Dk[k] @ V.T, 'fro') 
        if k >0:
            err = abs(value_of_n - value_of) 
            if err < tol: #Error
                break
        value_of = value_of_n
        iteration = i+1
    return U,Dk,V,err,iteration,value_of