import numpy as np
import time as tm

"""
    The function `calc_GAM` takes a list of matrices, calculates the GAM matrix for each matrix in the
    list, and returns a list containing all the calculated GAM matrices.
    
    :param matrixs: It looks like the function `calc_GAM` takes a list of matrices as input and
    calculates the Gramian matrix for each matrix in the list. The Gramian matrix is calculated as the
    matrix product of the transpose of the matrix with itself
    :return: The function `calc_GAM` takes a list of matrices as input, calculates the GAM (Generalized
    Autoregressive Model) matrix for each matrix in the list, and then returns a list containing all the
    calculated GAM matrices.
    """
def calc_GAM(matrixs):
    a_list =[]
    for i in matrixs:
        ai = i.T @ i #For each matrix in the list is calculated the GAM Matrix d_i ^T d_i
        a_list.append(ai) #Concatenate all the Gam matrix to the new list
    return a_list

"""
 Computes a symmetric matrix S from a list of matrices using pairwise
 combinations of each matrix and its pseudoinverse. Common in HO-SVD.

 Parameters:
 - matrixs (list): List of input matrices.
 - N (int): Number of matrices.

 Returns:
 - matrix_s (np.ndarray): Symmetric matrix result.
"""
def calc_S(matrixs, N):
    gam_list=calc_GAM(matrixs=matrixs)
    const = 1/(N*(N-1)) #Calculates the constant of the sumatory S =====> 1/N(N+1)
    suma =0
    for i in range(N):
        for j in range(i+1,N): #Here we ensure tha j always will be bigger than i and less than N 
          inv_j = np.linalg.pinv(gam_list[j]) #Inverts A_j
          inv_i = np.linalg.pinv(gam_list[i])#Inverts A_i 
          suma += (gam_list[i]@inv_j + gam_list[j] @ inv_i) #Here does the sumatory A_i A_j ^-1 + A_j A_i ^-1
    matrix_s = suma * const # Makes the product of the constant and the sumatorý
    return matrix_s
"""
    The function `calc_matrix_V` calculates the eigenvalues and eigenvectors of a given matrix.
    :param matrix_s: It looks like the function `calc_matrix_V` is designed to calculate the eigenvalues
    and eigenvectors of a given matrix `matrix_s`. The function returns the eigenvalues as a diagonal
    matrix `lamb` and the eigenvectors `V`
    :return: The function `calc_matrix_V` returns two values: `lamb`, which is a diagonal matrix
    containing the eigenvalues of the input matrix `matrix_s`, and `V`, which is a matrix containing the
    eigenvectors of the input matrix `matrix_s`.
    """
def calc_matrix_V(matrix_s):
    lamb, V = np.linalg.eig(matrix_s) #Calculates the eigen values and the eigenvectors of the matrix S
    lamb = np.diag(lamb)
    return V
"""
    The function `calc_B_list` takes two matrices as input, solves a system of linear equations using
    one matrix and then appends the solution to a list for each matrix in the second input matrix.
    
    :param matrix_v:A orthogonal matrix calculated in the last step
    :param matrix_D: Diagonal matrix
    parameter so that I can assist you further with the `calc_B_list` function?
    :return: The function `calc_B_list` returns a list of vectors `b_list`, where each vector is the
    result of solving a linear system of equations using the matrix `matrix_v` and the transpose of a
    matrix from the input `matrix_D`.
    """
def calc_B_list(matrix_v, matrix_D):
    b_list =[]
    for mat in matrix_D:
        xi = np.linalg.solve(matrix_v, mat.T) #Solve the ecuation bi = V Di.T
        bi = xi.T
        b_list.append(bi)
    return b_list
"""
    The function `calc_sigma` calculates the diagonal matrices of norms of column vectors in a list of
    matrices.
    
    :param b_list: The `b_list` parameter is a list of matrices. Each matrix in the list represents a
    set of vectors
    the function to iterate over the columns of each matrix in b_list
    :return: The function `calc_sigma` returns a list of diagonal matrices, where each diagonal matrix
    is constructed from the norms of the columns of the input matrices in `b_list`.
    """
def calc_sigma(b_list):
    sigma_list=[]
    for b_i in b_list:
        coef_vect =[]
        for i in range(b_i.shape[1]):
            coef = np.linalg.norm(b_i[:,i]) #Calculates the norm of every column of each matrix
            coef_vect.append(coef)
        sigma_list.append(np.diag(coef_vect)) #Creates a diagonal matrix with the norms in the diagonal
    return sigma_list
"""
    The function `calc_U_list` calculates a list of vectors `u_i` by solving a system of linear
    equations for each element in the input lists `b_list` and `sigma_list`.
    
    :param b_list: The `b_list` parameter is a list of matrices where each matrix represents a vector
    `b` for a system of linear equations
    :param sigma_list: matrix of singular values
    :param N: The parameter N represents the number of elements in the lists b_list and sigma_list. It
    is used as the range for the loop in the function calc_U_list to iterate over the elements in these
    lists
    :return: The function `calc_U_list` returns a list of vectors `u_list`, where each vector is the
    solution to the linear system `sigma_list[i].T * u_i = b_list[i].T` for `i` ranging from 0 to `N-1`.
    """
def calc_U_list(b_list, sigma_list,N):
    u_list= []
    for i in range(N):
        u_i = np.linalg.solve(sigma_list[i].T, b_list[i].T) #Solve the ecuation ui = Σi.T Bi.T
        u_list.append(u_i.T) # the result needs to be transposed
    return u_list
"""
    The function `verify_cols` checks if all matrices in a list have the same number of columns.
    :param mat_list: The `mat_list` parameter is a list of matrices. The function `verify_cols` iterates
    through the list of matrices and checks if the number of columns in each matrix is the same. If the
    number of columns is not the same for all matrices in the list, it raises a `Value
    """
def verify_cols(mat_list):
    for i in range(len(mat_list)-1):
        n1 = mat_list[i].shape[1]
        n2 = mat_list[i+1].shape[1]
        if n1 != n2:
            raise ValueError("The matrix columns must be the same for all the matrix in the array")
"""
    The function `high_Order_SVD` performs high-order singular value decomposition on a list of
    matrices.
    
    :param matrix_list: array of matrix with the same number of cols
    :return: The function `high_Order_SVD` is returning a tuple containing three elements:
    1. `matrix_u_list`: A list of matrices representing the left singular vectors.
    2. `sigma_list`: A list of singular values.
    3. `matrix_v`: The matrix representing the right singular vectors.
    """
def high_Order_SVD(matrix_list):
    verify_cols(matrix_list)
    N = len(matrix_list) #Takes de numbers of matrices to analize
    matrix_s = calc_S(matrix_list,N) #Calculates the matrix S, that is used to calculate de matrix V
    matrix_v = calc_matrix_V(matrix_s) 
    b_list = calc_B_list(matrix_v,matrix_list) #Calculates the list of matrices B, this matrices are used to calculate the sigma lists
    sigma_list = calc_sigma(b_list)
    matrix_u_list = calc_U_list(b_list,sigma_list,N) #calculates the list o matrices U
    return matrix_u_list, sigma_list, matrix_v 


