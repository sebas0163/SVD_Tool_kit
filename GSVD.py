import numpy as np

"""
    The function `householder_vector` computes the Householder vector and beta value for a given input
    vector.
    
    :param x: The `householder_vector` function takes a vector `x` as input and computes the Householder
    vector `v` and the scalar `beta` associated with it. The Householder transformation is a linear
    transformation that reflects a vector about a hyperplane defined by a unit vector
    :return: The function `householder_vector(x)` returns a Householder vector `v` and a scalar `beta`.
    """
def householder_vector(x):
    norm_x = np.linalg.norm(x)
    if norm_x == 0: #If the norm is 0 returns the vector without changes
        return x, 0
    e1 = np.zeros_like(x) #makes a vector with 0 execpt in the first position where is positionated the vector X's Norm
    e1[0] = norm_x * np.sign(x[0])
    v = x - e1 #Does the subtraction of both vectors
    v /= np.linalg.norm(v) # Normalize the vector 
    beta = 2 / np.dot(v, v) #Calculates Beta
    return v, beta
"""
    The `householder_qr` function in Python implements the Householder QR factorization algorithm to
    decompose a matrix A into the product of an orthogonal matrix Q and an upper triangular matrix R.
    
    :param A: The `householder_qr` function you provided implements the Householder QR factorization
    algorithm to decompose a matrix A into the product of an orthogonal matrix Q and an upper triangular
    matrix R
    :return: The function `householder_qr` returns the matrices Q and R, where Q is an orthogonal matrix
    of size m x m and R is an upper triangular matrix.
    """
def householder_qr(A):
    m, n = A.shape
    R = A.astype(float).copy() #Makes a copy of the original matrix
    Q = np.eye(m)  #Creates a new matrix with mxm dimentions
    for j in range(n):
        v, beta = householder_vector(R[j:, j]) #Asks for the new vector and the factor beta
        v_full = np.zeros(m)
        v_full[j:] = v #Add the new vector from the position j until the end
        R -= beta * np.outer(v_full, v_full @ R) #Applies house holder trnasformation
        Q -= beta * np.outer(Q @ v_full, v_full)
    return Q, R



#casos de Prueba
A = np.array([[4, 1], [3, -1], [5, 2]], dtype=float)
Q, R = householder_qr(A)
