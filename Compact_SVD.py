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


"""
def prueba():
    i = 100
    while i <=1000:
        matrix_ =np.random.rand(2*i,i)
        init_alg_1 =tm.perf_counter()
        #print(matrix_)
        ur,s,vr =compact_svd(matrix=matrix_)
        fin_alg_1 =tm.perf_counter()
        init_np_alg = tm.perf_counter()
        a,b,c =np.linalg.svd(matrix_)
        fin_np_alg = tm.perf_counter()

        print("Tiempo de algoritmo propio: " + str(i))
        print(fin_alg_1-init_alg_1)
        print("Tiempo de algoritmo numpy " + str(i))
        print(fin_np_alg-init_np_alg)
        i = i +100
"""
#==============Test====================================
#prueba()
#i = 100
#matrix_ =np.random.rand(i,2*i)
#print(matrix_)
# ur,s,vr =compact_svd(matrix=matrix_)
#aux=np.dot(ur,s)
#aux= np.dot(aux, vr)
#print("=========================================")
#print(aux)
    
#Notas de prueba
"""
1: Crear matrices de tamaño m x 2m, donde m=100:100:1000
2: Crear matrices de tamaño 2m x m, donde m=100:100:1000
3: Crear matrices de tamaño m x m, donde m=100:100:1000
Calcular el tiempo de ejecución en calcular la SVD de Numpy y la SVD compacta
Presentar una gráfica en cada experimento dimension m vs tiempo en segundos 
Método T1 metodo nuestro T2 metodo de numpy para obtener porcentaje de diferencia que tan eficiente es uno a otro  
Multiplicar para llegar a la misma matriz y ver documentación
"""