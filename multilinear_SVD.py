import numpy as np
from scipy.linalg import block_diag
from tensorly import unfold
#Funcionamiento correcto
def detect_Range(h_list):
    large = len(h_list)
    for i in range(large-1):
        if np.linalg.matrix_rank(h_list[i]) != np.linalg.matrix_rank(h_list[i+1]):
            return False
    return True
#funciona 
def calc_Q_J(h_list):
    # Calcula Q y J correctamente
    J = max([h.shape[0] for h in h_list])
    I = h_list[0].shape[1]
    Q = min(J, I)
    return J, Q
def frob_for_tensor(tensor):
    sumatory =0 
    for r in tensor:
        for row in r:
            for elm in row:
                sumatory += elm**2
    return np.sqrt(sumatory)
def error(h_tild, h_list):
    num = h_tild - h_list
    num = frob_for_tensor(num)
    den = frob_for_tensor(h_list)
    if den ==0:
        den = 0.00000001
    return (num**2)/(den**2)
def intialize(h_list,k,j,q,col_num): #mejorar inicializacion preguntar mañana
    comprobation = detect_Range(h_list)
    mat_A = np.random.rand(col_num, q)
    mat_C = np.abs(np.random.rand(k,q))
    return mat_A, mat_C
def mode_unfold(tensor, mode):
    tensor = np.array(tensor)
    if mode == 1:
        return tensor.transpose(1, 0, 2).reshape(tensor.shape[1], -1)
    elif mode == 2:
        return tensor.reshape(tensor.shape[0], -1)
    elif mode == 3:
        return tensor.transpose(2, 0, 1).reshape(tensor.shape[2], -1)
    else:
        raise ValueError("El modo debe ser 1, 2 o 3")
def hierarchical_sort(matrix):
    Q = matrix.shape[1]
    sorted_indices = np.arange(Q)  # Índices iniciales sin ordenar
    
    # Ordenar jerárquicamente fila por fila
    for i in range(matrix.shape[0]):
        sorted_indices = sorted_indices[np.argsort(matrix[i, sorted_indices])]  # Orden por fila i
    
    C_sorted = matrix[:, sorted_indices]  # Aplicar la ordenación a C
    return C_sorted, sorted_indices
def reorder_ACBk(C, A, B_list):
    C_sorted, sorted_indices = hierarchical_sort(C)
    A_sorted = A[:, sorted_indices]  
    B_list_sorted = [Bk[:, sorted_indices] for Bk in B_list]  
    return C_sorted, A_sorted, B_list_sorted
def multilineal_SVD(h_list):
    k = len(h_list)
    l= len(h_list[0])
    j,q = calc_Q_J(h_list)
    b_list =[]
    cota = 1.5
    iteration =0
    #creat la matriz random
    mat_A, mat_C = intialize(h_list, k,j,q,l)
    #Bucle principal
    while iteration < 1000: #modoficar
        #Bucle de reconstrucción de b_K
        h_sub_list =[]
        iteration +=1
        for i in range(k):
            h_k = h_list[i]
            h_k_sub = mat_A @ np.diag(mat_C[i,:])
            t_k = h_k @ h_k_sub
            b_k = np.linalg.pinv(((t_k @ t_k.T))**(1/2))@t_k
            h_k_sub = h_k.T @ b_k
            b_list.append(b_k)
            h_sub_list.append(h_k_sub) 
        #bucle de reconstrucción de A y C
        h_sub_list_m = np.array(h_sub_list)
        blocks =[]
        for i in range(q):
            denom = (np.linalg.norm(mat_C[:,i]))**2
            if denom == 0:
                denom = 1
            bloq = (mat_C[:,i].T)/denom #Here it's calculated the ecuatiion C(:,i)/norm(C(:,i))^2
            bloq = np.diag(bloq)
            blocks.append(bloq)
        b_diag = np.block([i for i in blocks])
        h_1 = mode_unfold(h_sub_list_m,1)
        print("bdiag\n",b_diag.shape)
        print("h\n",h_1.shape)
        mat_A = h_1@b_diag.T #Revisar que esta matriz sea correcta
        #Reconstrucción de C
        mat_c_aux =[]
        for i in range(q):
            denom = np.linalg.norm(mat_A[:,i])**2
            if denom ==0:
                denom = 1
            diag = np.diag(((mat_A[i,:].T) /denom))
            if len(mat_c_aux) ==0:
                mat_c_aux = np.array(diag)
            else:
                mat_c_aux = np.hstack((mat_c_aux,diag))
        h_3 = mode_unfold(h_sub_list_m,3)
        mat_C= h_3@ mat_c_aux.T
        #Normalizacion de las columnas de C
        norms = np.linalg.norm(mat_C,axis=0)
        norms[norms ==0] =1
        mat_C = mat_C / norms
        #Arreglar C para que sea no negativo =====> si hay menos se traslada a A esto por un bucle
        c_row, c_col = mat_C.shape
        for i in range(c_row): #Posible bug que A y C no son del mismo tamaño 
            for r in range(c_col): #modificar ya que puede que las filas sean distintas
                if mat_C[i][r] < 0:
                    mat_C[i][r] = abs(mat_C[i][r])
                    mat_A[i][r] = mat_C[i][r] *-1
        # Bucle para calcular h tilde
        h_k_tild =[]
        for i in range(k):
            h_k_tild.append(mat_A @ np.diag(c[i,:]) @ b_list[i].T)
        # calcular error
        h_k_tild = np.array(h_k_tild)
        h_list_m = np.array(h_list)
        err = error(h_k_tild, h_list_m)
        print("error \n",err)
        if err < cota:
            break
    #Ordenar las columnas de C A Y H 
    mat_C, mat_A, b_list =reorder_ACBk(mat_C,mat_A,b_list)
    return mat_C, mat_A, b_list

a = np.array([[1,2,1],[6,7,1],[10,12,1]])
b = np.array([[1,2,4],[6,7,8],[10,11,12]])
c = np.array([[1,2,3],[6,7,5],[10,11,13]])
h_ls = [a,b,c]
h_ls = np.array(h_ls)
mat_C, mat_A, b_list =multilineal_SVD(h_ls)

h_1 = b_list[0] @ np.diag(c[1,:]) @ mat_A.T
print(h_1)

"""
Frobenius suma de todos las entradas al cuadrado (fondo, filas,  columnas)
Construir con matrice predefinidas es decir definir A c_k y b_k para multiplicar ver el resultado
La diferencia de la matriz menos el producto de las tres matrices sea 0 
Que cumpla con las caracterisitcias.
"""