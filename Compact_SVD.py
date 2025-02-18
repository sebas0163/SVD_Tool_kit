import numpy as np

def get_S_matrix(eigenValues):
    new_eigenvalues = []
    for eigenval in eigenValues: #Calculate the square root of the eigenvalue different of zero
        if eigenval != 0:
            new_eigenvalues.append(np.sqrt(eigenval))
    new_eigenvalues.sort(reverse=True) # sort the eigenvalues from the min to the max
    rang = len(new_eigenvalues) # Takes the range of the matrix
    matrix_S = np.zeros((rang,rang))
    for index in range(rang):
        matrix_S[index][index] = new_eigenvalues[index] #Create a diagonal matrix with the eigen values
    return matrix_S
def get_Ur_Vr_Matrix(dot_product):
    eigenval, eigenvec = np.linalg.eig(dot_product)
    indx_no_null = np.where(eigenval != 0)[0]
    eigenvec = eigenvec[:,indx_no_null]
    return eigenvec

def compact_svd(matrix):
    #Calculate the matrix transpose
    trasp = matrix.transpose()
    dot_product = np.zeros((1,1))
    #Calculate the matrix's dimentions to take a decition about the next operation.
    if len(matrix) <= len(matrix[0]):
        dot_product = np.dot(matrix, trasp) #makes the dot product between A and A ^t
    else:
        dot_product = np.dot(trasp, matrix) #makes the dot product between A ^t and A
    #Calculate the eigenvalues
    eigenvalues = np.linalg.eigvals(dot_product)
    matrix_S = get_S_matrix(eigenvalues)
    #Calculate Ur or Vr
    if len(matrix) <= len(matrix[0]):
        ur_matrix = get_Ur_Vr_Matrix(dot_product) #Calculates the matrix Ur
        dimen_m = len(ur_matrix)
        vr_matrix = []
        for index in range(dimen_m):
            vj = (1/(matrix_S[index][index])) * np.dot(trasp,ur_matrix[:,index].reshape(-1,1))
            vr_matrix.append(np.transpose(vj)[0])
        vr_matrix = np.transpose(vr_matrix)
        vr_matrix = np.transpose(vr_matrix)
        print(ur_matrix)
        print("================== VR")
        print(vr_matrix)
        print("================== S")
        print(matrix_S)
    else:
        vr_matrix = get_Ur_Vr_Matrix(dot_product) #Calculates the matrix Ur
        dimen_m = len(vr_matrix[0])
        ur_matrix =[]
        for index in range(dimen_m):
            uj = (1/(matrix_S[index][index])) * np.dot(matrix,vr_matrix[:,index].reshape(-1,1))
            ur_matrix.append(uj)
        ur_matrix =np.array(ur_matrix)
        ur_matrix = np.transpose(ur_matrix)
        print(ur_matrix)
        print("================== Matrix VR")
        print(np.transpose(vr_matrix))
        print("================== Matrix S")
        print(matrix_S)
        #Add retorn 

matrix_ =np.array([[2,1,2],[2,-2,2],[-2,-1,-2],[2,0,2]])
compact_svd(matrix=matrix_)
    
