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


#========================================================Recursive BLOCK HOUSE HOLDER Implementation===================########### 
"""
    The function `recursive_block_qr` recursively computes the QR decomposition of a matrix using a
    block algorithm with a specified block size.
    
    :param A: The parameter A is a matrix that represents the input matrix for the QR decomposition
    :param n: The parameter `n` in the `recursive_block_qr` function represents the size of the matrix
    `A` along a particular dimension. It is used to determine the size of the matrix and to control the
    recursion depth in the QR decomposition algorithm
    :param nb: The parameter `nb` in the `recursive_block_qr` function represents the block size for the
    recursive QR decomposition algorithm. It determines the size of the blocks into which the matrix is
    divided during the recursive process. When the size of the block reaches or falls below `nb`, the QR
    decomposition is
    :return: The function `recursive_block_qr` returns two matrices `q` and `r`, which are the results
    of the QR decomposition of the input matrix `A`.
    """
def recursive_block_qr(A,n,nb):
    if n <= nb:
        q,r =householder_qr(A) #If the number of cols are less or equal to the limit asks for the normal house holder
    else:
        n_1 = int(np.floor(n/2))
        q_1,r_11 = recursive_block_qr(A[:,:n_1],n_1,nb) #Reduce the blocks 
        r_12 = q_1.T @ A[:,n_1:]
        A[:,n_1:] = A[:,n_1+1:n] - q_1 @ r_12
        q_2, r_22 = recursive_block_qr(A[:,n_1:],n-n_1, nb)
        q = np.hstack((q_1,q_2))
        r = np.vstack((np.hstack((r_11,r_12)), np.hstack((np.zeros((r_22.shape[0],r_11.shape[1])),r_22))))
    return q,r

#=================================Givens Qmethods===================#############3
"""
    The given function implements the Givens rotation method to calculate the values of c and s based on
    input parameters a and b.
    
    :param a: The parameter `a` in the `givens_rotation` function represents the value of the element in
    the matrix that will be used in the Givens rotation
    :param b: The parameter `b` in the `givens_rotation` function represents one of the elements in a 2D
    vector that you want to rotate
    :return: The function `givens_rotation` returns the values of `c` and `s` based on the input
    parameters `a` and `b`.
    """
def givens_rotation(a,b):
    if b ==0:
        c = 1
        s =0
    else: 
        if abs(b) > abs(a):
            tau = -a/b
            s = 1/ np.sqrt(1+ tau **2)
            c = s* tau
        else:
            tau = -b/a 
            c = 1 / np.sqrt(1+tau**2)
            s = c *tau
    return c,s
"""
    The function `givens_QR` performs QR decomposition on a given matrix using Givens rotations.
    
    :param A: It seems like the code snippet you provided is for computing the QR decomposition of a
    matrix using Givens rotations. However, you have not provided the matrix A for which you want to
    compute the QR decomposition. Please provide the matrix A so that I can assist you further
    :return: The function `givens_QR` returns the matrices Q and R, where Q is an orthogonal matrix and
    R is an upper triangular matrix, obtained from the QR decomposition of the input matrix A.
    """
def givens_QR(A):
    m,n = A.shape
    q = np.eye(m)
    r = A.copy()
    for j in range(n):
        for i in range(m-1,j,-1):
            c, s = givens_rotation(r[i-1,j],r[i,j])
            g = np.array([[c,s],[-s,c]])
            r[i-1:i+1,j:] =np.dot(g.T, r[i-1:i+1, j:])
            q[:,i-1:i+1] = np.dot(q[:,i-1:i+1],g)
    return q,r




A = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=float)
nb = 2  # Tamaño del bloque
q,r = np.linalg.qr(A, 'complete')
Q, R = givens_QR(A)#recursive_block_qr(A,A.shape[1], nb)
print("Q:\n", Q)
print("R:\n", R)
print("QR\n", Q @ R)
print("Q_numpy:\n", q)
print("R_numpy:\n", r)
print("QR_numpy\n", q @ r)