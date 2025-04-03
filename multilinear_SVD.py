import numpy as np

def get_Q_I(tensor):
    J = min([h.shape[0] for h in tensor])
    I = tensor[0].shape[1]
    Q = min(J, I)
    return Q,I
def mode_unfold(tensor, mode):
    tensor = np.array(tensor)
    if mode == 1:
        k,m,n = tensor.shape
        result =[]
        for col in range(n):
            for row in range(m):
                vect = []
                for mat in range(k):
                    vect.append(tensor[mat][row][col])
                if len(result) ==0:
                    result=np.array([vect]).T
                else:
                    vect = np.array([vect]).T
                    result = np.hstack((result,vect))
        return result        
    elif mode == 2:
        return tensor.reshape(tensor.shape[0], -1) #no es así 
    elif mode == 3:
        result =[]
        for mat in tensor:
            for row in mat:
                if len(result) ==0:
                    result = np.array([row]).T
                else:
                    result = np.hstack((result,np.array([row]).T))
        return result
    else:
        raise ValueError("El modo debe ser 1, 2 o 3")
def initialice(l,q,k):
    mat_A = np.random.randn(l,q)
    mat_C = np.random.randn(k,q)
    return mat_A, mat_C

def h_reconstruct(mat_A, mat_C,k,tensor):
    h_reconst =[]
    b_k_list=[]
    for i in range(k):
        h_k_tild = mat_A @ np.diag(mat_C[i,:])
        t_k = tensor[i] @ h_k_tild
        b_k = np.linalg.pinv(((t_k @ t_k.T))**(1/2))@t_k
        b_k_list.append(b_k)
        h_k_tild = tensor[i].T @ b_k
        h_reconst.append(h_k_tild)
    h_reconst = np.array(h_reconst)
    b_k_list = np.array(b_k_list)
    return h_reconst,b_k_list
    
def act_mat_A(mat_C,q,h_reconst):
    unfl_mode_1 = mode_unfold(h_reconst,1)
    bloq_list =[]
    for i in range(q):
        col = mat_C[:,i]
        denom = (np.linalg.norm(col))**2
        if denom ==0:
            denom = 0.00001
        bloq = np.diag((col.T)/denom)
        bloq_list.append(bloq)
    bloq_diag = np.block([b for b in bloq_list])
    mat_A = unfl_mode_1 @ bloq_diag.T
    return mat_A
def act_mat_C(mat_A, q, h_reconst):
    diag_list =[]
    unfld_mode_3 = mode_unfold(h_reconst,3)
    for i in range(q):
        num = mat_A[i,:].T
        col = mat_A[:,i]
        denom = (np.linalg.norm(col))**2
        if denom == 0:
            denom = 0.00000001
        diag = np.diag(num/denom)
        if i ==0:
            diag_list =np.array(diag)
        else:
            diag_list = np.hstack((diag_list,diag))
    mat_C = unfld_mode_3 @ diag_list.T
    return mat_C
def normalice_C(mat_C,q):
    for i in range(q):
        col = mat_C[:,i]
        norm = np.linalg.norm(col)
        if norm ==0:
            norm =0.0000001
        mat_C[:,i] = col / norm
    return mat_C
def abs_mat_C(mat_C, mat_A):
    m,n = mat_C.shape
    for i in range(m):
        for j in range(n):
            if mat_C[i][j] <0:
                mat_C[i][j] = abs(mat_C[i][j])
                mat_A[i][j] = mat_A[i][j] *-1
    return mat_A, mat_C
def act_h_reconst(mat_A, mat_C, k, b_k_list):
    h_reconst =[]
    for i in range(k):
        h_k_tild = mat_A @ np.diag(mat_C[i,:]) @ b_k_list[i].T
        h_reconst.append(h_k_tild)
    return np.array(h_reconst)
def frob_tensor(tensor):
    sumatory =0
    for mat in tensor:
        for row in mat:
            for elm in row:
                sumatory += elm**2
    return sumatory
def errorr(h_reconst, tensor):
    num = h_reconst - tensor
    num = frob_tensor(num)
    denom = frob_tensor(tensor)
    if denom ==0:
        denom = 0.0000001
    return num/denom
def multilineal_SVD(tensor):
    k = len(tensor)
    q, l = get_Q_I(tensor)
    mat_A, mat_C = initialice(l,q,k)
    cote =1
    iterations = 0
    err_old = 10
    while iterations < 1000:
        h_reconst, b_k_list = h_reconstruct(mat_A,mat_C,k,tensor)
        #Reconstruct A
        mat_A = act_mat_A(mat_C,q,h_reconst)
        #Reconst C
        mat_C = act_mat_C(mat_A,q,h_reconst)
        #Normalice colums of C
        mat_C = normalice_C(mat_C,q)
        #compensation
        mat_A, mat_C = abs_mat_C(mat_C,mat_A)
        #Act_H_k
        h_reconst = act_h_reconst(mat_A,mat_C,k,b_k_list)
        #Calc err
        err = errorr(h_reconst,tensor)
        delta_err = (err_old-err)/err_old
        err_old = err
        print(delta_err)
        if abs(delta_err) < cote:
            break
        iterations +=1
    return mat_A, mat_C, b_k_list

a = np.array([[1,2,1],[6,7,1],[10,12,1]])
b = np.array([[1,2,4],[6,7,8],[10,11,12]])
c = np.array([[1,2,3],[6,7,5],[10,11,13]])
lista = [a,b,c]
tensor = np.array(lista)
mat_A, mat_C, b_k_list = multilineal_SVD(tensor)
print("mat_a \n",mat_A.shape)
print("mat_c \n",np.diag(mat_C[0,:]))
print("mat_b_k\n", b_k_list[0].shape)
print(b_k_list[0]@np.diag(mat_C[0,:])@mat_A[0].T)
"""
Frobenius suma de todos las entradas al cuadrado (fondo, filas,  columnas)
Construir con matrice predefinidas es decir definir A c_k y b_k para multiplicar ver el resultado
La diferencia de la matriz menos el producto de las tres matrices sea 0 
Que cumpla con las caracterisitcias.
"""