import numpy as np
"""
    The function `get_S_matrix` filters, sorts, and transforms positive eigenvalues into a diagonal
    matrix.
    
    :param eigenValues: The `eigenValues` parameter in the `get_S_matrix` function represents a list of
    eigenvalues. The function first filters out any negative eigenvalues, then sorts the remaining
    positive eigenvalues in descending order, and finally constructs a diagonal matrix using these
    sorted eigenvalues
    :return: The function `get_S_matrix` returns a diagonal matrix created from the positive square root
    of the input eigenValues array. The eigenValues array is first filtered to keep only the positive
    values, then sorted in descending order. Finally, these sorted positive square root values are used
    to create a diagonal matrix, which is then returned by the function.
    """
def get_S_matrix(eigenValues): #Optimized code for de matrix S
    eigenValues = np.sqrt(eigenValues[eigenValues>0])  #filter and sort the eigenvalues 
    eigenValues[::-1].sort()
    return np.diag(eigenValues) #transform de eigenvalues in diagonal matrix 
"""
    The function `get_Ur_Vr_Matrix` calculates the eigenvectors of a given dot product matrix and
    returns only the eigenvectors corresponding to positive eigenvalues.
    
    :param dot_product: The `dot_product` parameter in the `get_Ur_Vr_Matrix` function is typically a
    matrix representing the dot product of two vectors. The function calculates the eigenvalues and
    eigenvectors of this matrix and then filters out the eigenvectors corresponding to non-zero
    eigenvalues before returning them
    :return: The function `get_Ur_Vr_Matrix` returns the eigenvectors corresponding to the eigenvalues
    greater than 0 from the input dot product matrix.
    """
def get_Ur_Vr_Matrix(dot_product):
    eigenval, eigenvec = np.linalg.eigh(dot_product) #Get the eigenvalues and the eigenvectors
    return eigenvec[:,eigenval>0] #Filter the eigenvects diferents of zero
"""
    The function `compact_svd` calculates the compact singular value decomposition (SVD) of a given
    matrix.
    
    :param matrix: The function `compact_svd` you provided seems to be implementing the Singular Value
    Decomposition (SVD) algorithm. However, there are some functions like `get_S_matrix` and
    `get_Ur_Vr_Matrix` that are not defined in the code snippet you provided
    :return: The `compact_svd` function returns three matrices: `ur_matrix`, `matrix_S`, and
    `vr_matrix`. These matrices represent the left singular vectors, singular values, and right singular
    vectors respectively, obtained from the Singular Value Decomposition (SVD) of the input matrix.
    """
def compact_svd(matrix):
    m,n = matrix.shape #get the matrix's dimentions
    #Calculate if  the matrix transpose is needed 
    transpose_order = m>n
    #makes the dot product of A A^T or A^T. dependes of the dimentions
    dot_product = np.dot(matrix,matrix.T) if not transpose_order else np.dot(matrix.T,matrix)
    #Calculate the Matrix S and the eigenvalues
    eigenvalues = np.linalg.eigvalsh(dot_product)
    matrix_S = get_S_matrix(eigenvalues)
    #Calculates de eigenvects of the Ur matrix or Vr matrix
    eigenvects = get_Ur_Vr_Matrix(dot_product)
    if not transpose_order:
        ur_matrix = eigenvects
        vr_matrix = np.dot(matrix.T, ur_matrix) / np.diag(matrix_S) #Calculates the vectors Vj = 1/singval A.T uj
    else:
        vr_matrix = eigenvects
        ur_matrix = np.dot(matrix, vr_matrix) / np.diag(matrix_S)  #Calculates the vectors uj = 1/singval A vj
    return ur_matrix, matrix_S, vr_matrix