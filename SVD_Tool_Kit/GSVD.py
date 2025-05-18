import numpy as np
import time as tm
from scipy.linalg import cossin
from .SVD import svd

"""
    The GSVD function calculates the generalized singular value decomposition of two input matrices and
    returns specific components of the decomposition.
    
    :param matrix_A: Matrix of m_1xn size
    :param matrix_B: Matrix of m_2xn size
    :return: The function `GSVD` returns the following variables:
    - `u_1`: Matrix U1 calculated from the SVD decomposition
    - `u_2`: Matrix U2 calculated from the SVD decomposition
    - `d_a`: Matrix D_A extracted from the CS decomposition
    - `d_b`: Matrix D_B extracted from the CS decomposition
    - `x`: Matrix X calculated based on
    """
def GSVD(matrix_A,matrix_B):
    m,n = matrix_A.shape
    j,k = matrix_B.shape
    if k != n: #Validate the number of columns
        raise ValueError("Number of columns between matrix A and B are different")
    else:
        stacked_matrix = np.vstack((matrix_A,matrix_B))
        rank = np.linalg.matrix_rank(stacked_matrix)
        q, s,z =np.linalg.svd(stacked_matrix,full_matrices=True) #Calculates the SVD vals (change for my implementation complete)
        sigma_r = np.diag(s[:rank])#Here works well
        u,cs,vh = cossin(q, p=m, q=rank) #Here makes de CS decomposition for matrix Q obtained in the svd p is the number of the first m rows of the submatrix q11 and q is the rank
        u_1 = u[:m,:rank] #Calculates U1 from u
        u_2 =u[m:,rank:] #Calculates U2 from u
        d_a = cs[:rank, :rank] #Takes de d_a matrix from CS matrix
        d_b = cs[rank:,:rank] #Takes de d_a matrix from CS matrix tiene un bug
        v_1 = vh[:n,:n] @ sigma_r #Calculates v1 from vh matrix and multiplies by sigma 
        im = np.eye(n-rank) #Generates a Identity matrix with dimentions n-rank x n-rank
        zero_block_1 = np.zeros((rank,n-rank)) #Create a zero block to complete the new matrix
        zero_block_2 = np.zeros((n-rank, rank))#Create a zero block to complete the new matrix
        top_block = np.hstack((v_1,zero_block_1))
        bottom_block = np.hstack((zero_block_2,im))
        combined_matrix = np.vstack((top_block,bottom_block))#Creates the new coombined matrix
        x = z.T @ np.linalg.inv(combined_matrix) #Calculates x doing Z dot (combined_matrix)^-1
        return u_1, u_2, d_a,d_b, x #A is calculated with = u_1 @ d_a @ np.linalg.inv(x) B_exp = u_2@d_b @ np.linalg.inv(x)