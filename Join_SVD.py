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
def start():
    m, n, K = 64, 100, 10
    # U1, _ = np.linalg.qr(np.random.rand(m, m))
    # V1, _ = np.linalg.qr(np.random.rand(n, n))
    A = np.random.rand(K, m, n)
    #for k in range(K):
    #    A[k] = U1 @ diag_rec(np.random.rand(m, n)) @ V1.T
    return A
"""
    The function `join_SVD` performs a joint singular value decomposition on a set of matrices to find
    the low-rank approximation.
    
    :param matrix_set: It seems like the code snippet you provided is a Python function for performing
    Singular Value Decomposition (SVD) on a set of matrices stored in `matrix_set`. The function
    iteratively calculates the SVD of each matrix in the set and updates the singular vectors `U` and
    `V` until
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
            value_of_n += np.linalg.norm(matrix_set[k] - U @ Dk[k] @ V.T, 'fro') #esto va a ser valor de la función objetivo
        #print(value_of_n)
        if k >0:
            err = abs(value_of_n - value_of) #criterio de parada
            #print(f"Iter {i+1}, error: {err:.2e}")
            if err < tol: #Error
                print(f"Converged at iteration {i+1}")
                break
        value_of = value_of_n
    print(value_of)
    print(err)
    return U,Dk,V
"""
Para la compact SVD se tiene
Elegir el algoritmo que quiera del paper
HO para eliminar ruido
Tensor compresión pero aplicarlo a una imagen a color RGB 
Quateriniones compresión con color
Joint resolver sistema de ecuaciones 
"""
A_=start()
join_SVD(A_)
