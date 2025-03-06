import numpy as np
import time as tm
def get_S_matrix(eigenValues): #Optimized code for de matrix S
    eigenValues = np.sqrt(eigenValues[eigenValues>0])  #filter and sort the eigenvalues 
    eigenValues[::-1].sort()
    return np.diag(eigenValues) #transform de eigenvalues in diagonal matrix 
def get_Ur_Vr_Matrix(dot_product):
    eigenval, eigenvec = np.linalg.eigh(dot_product) #Get the eigenvalues and the eigenvectors
    return eigenvec[:,eigenval>0] #Filter the eigenvects diferents of zero

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
    return ur_matrix, matrix_S, vr_matrix.T