import numpy as np
import time as tm
from LU_fact import lu_factorization,backward_substitution,forward_substitution

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
    This function calculates the sum of products of inverses of matrices in a given list and returns the
    result multiplied by a constant.
    
    :param matrixs: It seems like the description of the `matrixs` parameter is incomplete. Could you
    please provide more information on what the `matrixs` parameter represents or what kind of data it
    contains? This will help in understanding the context of the function `calc_S` and how it operates
    on the input data
    :param N: It seems like the function `calc_S` is designed to calculate a matrix `matrix_s` based on
    the input parameters `matrixs` and `N`. However, the definition of the parameter `N` is missing.
    Could you please provide more information about what `N` represents in this context
    :return: The function `calc_S` returns the matrix `matrix_s`, which is the result of the calculation
    involving the input matrices `matrixs` and the constant `const`.
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
    
    :param matrix_v: It seems like the description of the `matrix_v` parameter is missing. Could you
    please provide more information about what `matrix_v` represents or how it is structured so that I
    can assist you further with the `calc_B_list` function?
    :param matrix_D: It seems like you were about to provide some information about the `matrix_D`
    parameter but the message got cut off. Could you please provide more details about the `matrix_D`
    parameter so that I can assist you further with the `calc_B_list` function?
    :return: The function `calc_B_list` returns a list of vectors `b_list`, where each vector is the
    result of solving a linear system of equations using the matrix `matrix_v` and the transpose of a
    matrix from the input `matrix_D`.
    """
def calc_B_list(matrix_v, matrix_D):
    L,U = lu_factorization(matrix_v)
    b_list =[] # cambiar por fact LU para V =====> LU = V x===> bi.T y y ====> UX
    for mat in matrix_D:
        xi =[]
        for col in range(mat.shape[1]):
            y = forward_substitution(L, mat[:,col])
            xi_k = backward_substitution(U,y)
            if col == 0:
               xi = np.array(xi_k.reshape(-1,1))
            else:
                xi = np.column_stack((xi,xi_k.reshape(-1,1)))
        bi = xi.T #N 
        b_list.append(bi)
    return b_list #luchazam
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
    :param sigma_list: It seems like you were about to provide more information about the parameters,
    but the message got cut off. Could you please provide more details about the `sigma_list` parameter
    so that I can assist you further with the `calc_U_list` function?
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
    The function `high_Order_SVD` performs high-order singular value decomposition on a list of
    matrices.
    
    :param matrix_list: It seems like the description of the `matrix_list` parameter is missing. Could
    you please provide more information about what the `matrix_list` contains or represents so that I
    can assist you further with the `high_Order_SVD` function?
    :return: The function `high_Order_SVD` is returning a tuple containing three elements:
    1. `matrix_u_list`: A list of matrices representing the left singular vectors.
    2. `sigma_list`: A list of singular values.
    3. `matrix_v`: The matrix representing the right singular vectors.
    """
def high_Order_SVD(matrix_list):
    N = len(matrix_list) #Takes de numbers of matrices to analize
    matrix_s = calc_S(matrix_list,N) #Calculates the matrix S, that is used to calculate de matrix V
    matrix_v = calc_matrix_V(matrix_s) 
    b_list = calc_B_list(matrix_v,matrix_list) #Calculates the list of matrices B, this matrices are used to calculate the sigma lists
    sigma_list = calc_sigma(b_list)
    matrix_u_list = calc_U_list(b_list,sigma_list,N) #calculates the list o matrices U
    return matrix_u_list, sigma_list, matrix_v 
"""     
def prueba(i,j,N):
    matrix_list =[]
    for r in range(N):
        a =np.random.rand(i,j)
        matrix_list.append(a)
    ini = tm.perf_counter()
    m,l,o=high_Order_SVD(matrix_list=matrix_list)
    end = tm.perf_counter()
    org =matrix_list[0]
    exp=m[0]@l[0]@o.T
    mae =np.mean(np.abs(org-exp))
    err = (mae/np.mean(org))*100
    print("err: ",err )
    print(end-ini)
def prueba_more_rows(i,j,N):
    matrix_list =[]
    for r in range(N):
        a =np.random.rand(np.random.randint(j,i),j)
        matrix_list.append(a)
    ini = tm.perf_counter()
    m,l,o=high_Order_SVD(matrix_list=matrix_list)
    end = tm.perf_counter()
    org =matrix_list[0]
    exp=m[0]@l[0]@o.T
    mae =np.mean(np.abs(org-exp))
    err = (mae/np.mean(org))*100
    print("err: ",err )
    print(end-ini)
"""

#prueba(500,500,5)
#prueba_more_rows(100,10,5)


