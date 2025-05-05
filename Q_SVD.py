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
    mat_B.real = -mat_B.real #Erro in the example, with this it's the same example
    mat_A_bar = np.conjugate(mat_A)
    mat_B_bar = (mat_B * -1).conj()
    top = np.hstack((mat_A,mat_B_bar))
    bottom = np.hstack((mat_B, mat_A_bar))
    mat_Q_eq = np.vstack((top,bottom)) 
    return mat_Q_eq 
"""
    The function `get_S_Q` takes a vector of singular values, sorts them in descending order, removes
    duplicates within a tolerance, and returns a diagonal matrix with the unique singular values.
    
    :param vect_s: It seems like the code snippet you provided is a Python function that takes a list of
    singular values `vect_s` as input and returns a diagonal matrix with unique singular values
    :return: The function `get_S_Q` returns a diagonal matrix created from the unique singular values in
    the input vector `vect_s`. The function first sorts the singular values in descending order, then
    iterates through the sorted values to identify unique singular values based on a specified tolerance
    level `tol`. Finally, it constructs a diagonal matrix using these unique singular values and returns
    it.
    """
def get_S_Q(vect_s):
    tol=1e-8
    # Ordenar valores singulares de mayor a menor
    sorted_vals = sorted(vect_s, reverse=True)
    
    # Lista para almacenar valores únicos
    unique_vals = []

    for val in sorted_vals:
        # Si no hay valores o la diferencia relativa es significativa, se agrega
        if all(abs(val - u)/max(abs(u), abs(val), 1e-12) > tol for u in unique_vals):
            unique_vals.append(val)

    # Crear matriz diagonal
    return np.diag(unique_vals)
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
    mat_U, mat_S, mat_V = np.linalg.svd(mat_Q_eq,full_matrices=True)
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
    mat_U_Q = complex_mat_to_quat_mat(mat_U_Q_a,mat_U_Q_b)
    mat_V_Q = complex_mat_to_quat_mat(mat_V_Q_a,mat_V_Q_b)
    return mat_U_Q, mat_V_Q, mat_S
"""
    The function `dot_product_quat` calculates the dot product of two matrices represented as
    quaternions.
    
    :param mat_A: It seems like the definition of the `dot_product_quat` function is incomplete. Could
    you please provide the missing part of the code related to the `mat_A` parameter so that I can
    better understand the function and assist you further?
    :param mat_B: It seems like you have provided the function `dot_product_quat` that calculates the
    dot product of two matrices represented as quaternions. However, you have not provided the
    definition or content of `mat_B`. Could you please provide the content of `mat_B` so that I can
    assist you
    :return: The `dot_product_quat` function is returning the result of the dot product operation
    between two matrices `mat_A` and `mat_B`. The result is a new matrix of quaternions where each
    element is calculated by multiplying and summing the corresponding elements of `mat_A` and `mat_B`
    according to the rules of matrix multiplication.
    """
def dot_product_quat(mat_A, mat_B):
    m1, n1 = mat_A.shape
    m2, n2 = mat_B.shape
    if n1 != m2:
        raise ValueError("Inconsistent dimentions")
    result = np.zeros((m1, n2),dtype=quaternion.quaternion)
    for i in range(m1):  # Filas de A
        for j in range(n2):  # Columnas de B
            for k in range(n1):  # Elementos comunes
                result[i][j] += mat_A[i, k] * mat_B[k, j]
    return result 
"""
    The function calculates the Frobenius norm for a matrix of quaternions.
    
    :param mat: It seems like the `mat` parameter is a matrix represented as a list of lists. Each inner
    list represents a row in the matrix, and each element in the inner list represents an element in
    that row
    :return: The function `frob_for_quaternions` is returning the Frobenius norm of the input matrix
    `mat` treated as a collection of quaternions. The Frobenius norm of a matrix is calculated by taking
    the square root of the sum of the squares of all the elements in the matrix. In this case, the
    elements of the matrix are treated as quaternions, and the
    """
def frob_for_quaternions(mat):
    sumatory =quaternion.quaternion(0,0,0,0)
    for row in mat:
        for elm in row:
            sumatory+= elm**2
    return sumatory ** (1/2)