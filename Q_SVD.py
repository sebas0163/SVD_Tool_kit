import numpy as np
import quaternion

def quat_to_complex(quaternion_mat):
    mat_A = np.zeros(quaternion_mat.shape, dtype=np.complex128)
    mat_B = np.zeros(quaternion_mat.shape, dtype=np.complex128)
    for i in range(quaternion_mat.shape[0]):
        for j in range(quaternion_mat.shape[1]):
            q = quaternion_mat[i,j]
            alpha = q.real + q.x *1j
            beta = q.y +q.z *1j
            mat_A[i,j] = alpha
            mat_B[i,j] = beta
    return mat_A,mat_B

def calculate_Q_e(quaternion_mat):
    mat_A, mat_B = quat_to_complex(quaternion_mat)
    mat_B.real = -mat_B.real #Posible error incoherencia con la teoría
    mat_A_bar = np.conjugate(mat_A)
    mat_B_bar = np.conjugate(-mat_B)
    top = np.hstack((mat_A,mat_B_bar))
    bottom = np.hstack((mat_B, mat_A_bar))
    mat_Q_eq = np.vstack((top,bottom)) 
    return mat_Q_eq #Funciona con el ejemplo 
#Hasta aquí el ejemplo el paso uno funciona bien 

def find_val(vect, val):
    for i in vect:
        if i - val <0.1:
            return False
    return True

def get_S_Q(vect_s):
    vect =[]
    for i in vect_s:
        if find_val(vect, i):
           vect.append(i)
    return vect
def calc_complex_svd(mat_Q_eq):
    mat_U, mat_S, mat_V = np.linalg.svd(mat_Q_eq)
    return mat_U,mat_S, mat_V.T
def get_Uq_Vq(mat_U_eq, mat_V_eq):
    n, _ = mat_U_eq.shape
    n = n//2 
    u_1 = mat_U_eq[:n,:n]
    u_2 = mat_U_eq[n:,:n]
    v_1 = mat_V_eq[:n,:n]
    v_2 = mat_V_eq[n:,:n]
    mat_U_Q = np.empty((n,n), dtype=np.quaternion)
    mat_V_Q = np.empty((n,n), dtype=np.quaternion)
    for i in range(n):
        for j in range(n):
            alpha_u = u_1[i,j]
            alpha_v = v_1[i,j]
            beta_u = u_2[i,j]
            beta_v = v_2[i,j]
            mat_U_Q[i,j] = quaternion.quaternion(alpha_u.real, alpha_u.imag, beta_u.real, beta_u.imag)
            mat_V_Q[i,j] = quaternion.quaternion(alpha_v.real, alpha_v.imag, beta_v.real, beta_v.imag)
    a,b=quat_to_complex(mat_V_Q)
    print(a)
def Q_SVD(mat_Q):
    mat_Q_eq=calculate_Q_e(mat_Q)
    mat_U,vect_S,mat_V = calc_complex_svd(mat_Q_eq)
    vect_s_q = get_S_Q(vect_S)
    get_Uq_Vq(mat_U,mat_V)



q11 = quaternion.quaternion(1,1,1,1)
q12 = quaternion.quaternion(2,1,0,-1)
q21 = quaternion.quaternion(1,0,-1,2)
q22 = quaternion.quaternion(3,2,-2,1)
mat_q = np.array([[q11,q12],[q21,q22]])
Q_SVD(mat_q)