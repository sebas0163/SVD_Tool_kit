import numpy as np
import quaternion
"""
    The function `quat_to_complex` converts a matrix of quaternions into two matrices of complex
    numbers.
    
    :param quaternion_mat: It seems like the definition of the `quaternion_mat` parameter is missing in
    the provided code snippet. Could you please provide the definition or structure of the
    `quaternion_mat` parameter so that I can assist you further with the `quat_to_complex` function?
    :return: The function `quat_to_complex` returns two complex matrices `mat_A` and `mat_B` which are
    derived from the input quaternion matrix `quaternion_mat`. `mat_A` contains the real and imaginary
    parts of the quaternion elements, while `mat_B` contains the other two imaginary parts of the
    quaternion elements.
    """
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
"""
    The function `complex_mat_to_quat_mat` converts two complex matrices into a quaternion matrix
    element-wise.
    
    :param A: It seems like you were about to provide some information about the parameter A, but the
    text got cut off. Could you please provide more details about the parameter A so that I can assist
    you further with the `complex_mat_to_quat_mat` function?
    :param B: It seems like you were about to provide some information about the parameter B in the
    function `complex_mat_to_quat_mat`, but the information is missing. Could you please provide the
    necessary details about the parameter B so that I can assist you further?
    :return: The function `complex_mat_to_quat_mat` returns a matrix `Q` where each element is a
    quaternion created from the corresponding elements of matrices `A` and `B`.
    """
def complex_mat_to_quat_mat(A, B):
    assert A.shape == B.shape, "Las matrices A y B deben tener la misma forma"
    Q = np.empty(A.shape, dtype=np.quaternion)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            alpha = A[i, j]  # a + bi
            beta = B[i, j]   # c + di
            Q[i, j] = quaternion.quaternion(alpha.real, alpha.imag, beta.real, beta.imag)
    return Q
"""
    The function `calculate_Q_e` takes a quaternion matrix, converts it to complex form, and constructs
    a matrix based on the input.
    
    :param quaternion_mat: It looks like the function `calculate_Q_e` is designed to calculate a matrix
    `mat_Q_eq` based on the input quaternion matrix `quaternion_mat`. However, the function
    `quat_to_complex` is referenced but not defined in the provided code snippet
    :return: The function `calculate_Q_e` returns a matrix `mat_Q_eq` that is constructed by stacking
    two matrices `top` and `bottom` horizontally and then vertically. The `top` matrix is formed by
    horizontally stacking `mat_A` and the conjugate of `-mat_B`, while the `bottom` matrix is formed by
    horizontally stacking `mat_B` and the conjugate of `mat_A
    """
def calculate_Q_e(quaternion_mat):
    mat_A, mat_B = quat_to_complex(quaternion_mat)
    #mat_B.real = -mat_B.real #Erro in the example, with this it's the same example
    mat_A_bar = np.conjugate(mat_A)
    mat_B_bar = (mat_B * -1).conj()
    top = np.hstack((mat_A,mat_B_bar))
    bottom = np.hstack((mat_B, mat_A_bar))
    mat_Q_eq = np.vstack((top,bottom)) 
    return mat_Q_eq 
"""
    The function `find_val` checks if any element in the input vector is within 0.1 of a specified
    value.
    
    :param vect: The `vect` parameter is a list of values that the function `find_val` will iterate
    through to compare with the `val` parameter
    :param val: The `val` parameter in the `find_val` function represents the value that each element in
    the `vect` list is compared against. The function checks if the difference between each element in
    the `vect` list and the `val` parameter is less than 0.1. If any element
    :return: The function `find_val` is returning a boolean value. It returns `False` if any element in
    the `vect` list is within 0.1 of the `val` parameter, otherwise it returns `True`.
    """
def find_val(vect, val): #Deletes copy eigenvalues
    for i in vect:
        if i - val <0.1:
            return False
    return True
"""
    The function `get_S_Q` takes a vector as input, removes duplicate values, and returns a diagonal
    matrix with the unique values.
    
    :param vect_s: It seems like the definition of the function `get_S_Q` is incomplete. The function is
    trying to create a diagonal matrix using unique values from the input vector `vect_s`. However, the
    function is using a `find_val` function which is not defined in the provided code snippet
    :return: The function `get_S_Q` is returning a diagonal matrix created from the unique elements in
    the input vector `vect_s`.
    """
def get_S_Q(vect_s):
    vect =[]
    for i in vect_s:
        if find_val(vect, i):
           vect.append(i)
    return np.diag(vect)
"""
    The function `calc_complex_svd` calculates the singular value decomposition (SVD) of a complex
    matrix.
    
    :param mat_Q_eq: It looks like the function `calc_complex_svd` is designed to calculate the Singular
    Value Decomposition (SVD) of a complex matrix `mat_Q_eq`. The function uses NumPy's `linalg.svd`
    function to compute the SVD
    :return: The function `calc_complex_svd` returns three matrices: `mat_U`, `mat_S`, and the conjugate
    transpose of `mat_V`.
    """
def calc_complex_svd(mat_Q_eq):
    mat_U, mat_S, mat_V = np.linalg.svd(mat_Q_eq)
    return mat_U,mat_S, mat_V.conj().T
"""
    The function `get_Uq_Vq` takes two matrices as input, splits them into submatrices, and returns four
    submatrices based on specific conditions.
    
    :param mat_U_eq: It seems like you were about to provide some information about the `mat_U_eq`
    parameter, but the message got cut off. Could you please provide more details or complete the
    information so that I can assist you further with the `get_Uq_Vq` function?
    :param mat_V_eq: It seems like you have provided the code snippet without completing the description
    of the parameters. Could you please provide more information about the `mat_V_eq` parameter so that
    I can assist you further with the function `get_Uq_Vq`?
    :return: The function `get_Uq_Vq` returns four matrices: `mat_u_q_a`, `mat_u_q_b`, `mat_v_q_a`, and
    `mat_v_q_b`.
    """
def get_Uq_Vq(mat_U_eq, mat_V_eq):
    n = mat_U_eq.shape[1]
    j = len(mat_U_eq[:,0])
    mid = j //2
    mat_u_q_a = np.zeros((mid,mid),dtype=complex)
    mat_u_q_b = np.zeros((mid,mid),dtype=complex)
    mat_v_q_a = np.zeros((mid,mid),dtype=complex)
    mat_v_q_b = np.zeros((mid,mid),dtype=complex)
    r =0
    for i in range(0,n,2): #Itereate over every matrix but just in even-numbered colums 
        mat_u_q_a[:,r] = mat_U_eq[:mid,i]
        mat_u_q_b[:,r] = (mat_U_eq[mid:,i]*-1).conj() #The second bloq of the column must be the real part negated
        mat_v_q_a[:,r] = mat_V_eq[:mid,i]
        mat_v_q_b[:,r] = (mat_V_eq[mid:,i]*-1).conj()
        r +=1
    return mat_u_q_a,mat_u_q_b,mat_v_q_a,mat_v_q_b
"""
    The function Q_SVD performs Singular Value Decomposition (SVD) on a given matrix mat_Q and returns
    the quaternion matrices mat_U_Q, mat_V_Q, and the singular values matrix mat_S.
    
    :param mat_Q: It looks like the code snippet you provided is a function `Q_SVD` that performs some
    operations on a matrix `mat_Q` and returns transformed matrices `mat_U_Q`, `mat_V`, and `mat_S`
    :return: The function `Q_SVD` returns the quaternion matrices `mat_U_Q`, `mat_V_Q`, and the singular
    values matrix `mat_S`.
    """
def Q_SVD(mat_Q):
    m,n = mat_Q.shape
    if m !=n:
        raise ValueError("mat_Q must be square")
    mat_Q_eq=calculate_Q_e(mat_Q)
    mat_U,vect_S,mat_V = calc_complex_svd(mat_Q_eq)
    mat_S = get_S_Q(vect_S)
    mat_U_Q_a, mat_U_Q_b, mat_V_Q_a, mat_V_Q_b=get_Uq_Vq(mat_U,mat_V)
    print(mat_U_Q_a)
    mat_U_Q = complex_mat_to_quat_mat(mat_U_Q_a,mat_U_Q_b)
    mat_V_Q = complex_mat_to_quat_mat(mat_V_Q_a,mat_V_Q_b)
    return mat_U_Q, mat_V_Q, mat_S
    
q11 = quaternion.quaternion(1,1,1,1)
q12 = quaternion.quaternion(2,1,0,-1)
q21 = quaternion.quaternion(1,0,-1,2)
q22 = quaternion.quaternion(3,2,-2,1)
mat_q = np.array([[q11,q12],[q21,q22]])
Q_SVD(mat_q)