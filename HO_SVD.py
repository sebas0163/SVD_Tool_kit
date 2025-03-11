import numpy as np

def calc_GAM(matrixs):
    a_list =[]
    for i in matrixs:
        ai = i.T @ i #For each matrix in the list is calculated the GAM Matrix d_i ^T d_i
        a_list.append(ai) #Concatenate all the Gam matrix to the new list
    return a_list
def calc_S(matrixs, N):
    gam_list=calc_GAM(matrixs=matrixs)
    const = 1/(N*(N-1)) #Calculates the constant of the sumatory S =====> 1/N(N+1)
    suma =0
    for i in range(N):
        for j in range(i+1,N): #Here we ensure tha j always will be bigger than i and less than N 
          inv_j = np.linalg.inv(gam_list[j]) #Inverts A_j
          inv_i = np.linalg.inv(gam_list[i])#Inverts A_i 
          suma += (gam_list[i]@inv_j + gam_list[j] @ inv_i) #Here does the sumatory A_i A_j ^-1 + A_j A_i ^-1
    matrix_s = suma * const # Makes the product of the constant and the sumatorý
    return matrix_s
def calc_matrix_V(matrix_s):
    lamb, V = np.linalg.eig(matrix_s) #Calculates the eigen values and the eigenvectors of the matrix S
    lamb = np.diag(lamb)
    return lamb, V
def calc_S_i_list(matrix_list, matrix_v):
    n= matrix_v.shape[1]
    s_i_list =[] 
    for mat in matrix_list:
        s_i_vect =[]
        for i in range(n):
            col = matrix_v[:,i]
            s_i = mat @ col
            s_i = np.linalg.norm(s_i)
            s_i_vect.append(s_i)
        s_i_mat = np.diag(s_i_vect)
        s_i_list.append(s_i_mat)
    return s_i_list
def calc_Ui_list(matrix_list, matrix_v, matrix_s_list,N):
    u_list=[]
    for i in range(N):
        s_inv = np.linalg.inv(matrix_s_list[i])
        u_i = matrix_list[i]@matrix_v@s_inv
        u_i = np.linalg.norm(u_i)
        u_list.append(u_i)
    return u_list

def high_Order_SVD(matrix_list):
    N = len(matrix_list)
    matrix_s = calc_S(matrix_list,N)
    lamb,matrix_v = calc_matrix_V(matrix_s)
    matrix_s_list =calc_S_i_list(matrix_list,matrix_v)
    matrix_u_list = calc_Ui_list(matrix_list, matrix_v,matrix_s_list,N)
    print(matrix_u_list)

        


a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([[12,4,3],[45,57,65],[75,8,97],[12,13,14]])
c = np.array([[10,247,34],[4,54,645]])
d_list = [a,b,c]
high_Order_SVD(d_list)