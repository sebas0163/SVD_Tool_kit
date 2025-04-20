import numpy as np
from SVD import svd
"""
    The function `t_product` calculates the element-wise product of two 3D arrays after performing Fast
    Fourier Transform operations.
    
    :param t1: It seems like the description of the `t1` parameter is missing. Could you please provide
    more information about the `t1` tensor, such as its shape or contents, so that I can assist you
    further with the `t_product` function?
    :param t2: It seems like you have provided the code snippet for a function `t_product` that performs
    a tensor product operation using Fast Fourier Transform (FFT). However, the definition of the `t2`
    parameter is missing in your message. Could you please provide the definition or values of the `t2`
    :return: The function `t_product` returns the result of the element-wise multiplication of the
    Fourier transforms of two input tensors `t1` and `t2`. The function performs the multiplication in
    the frequency domain using FFT (Fast Fourier Transform) and then returns the inverse Fourier
    transform of the result.
    """
def t_product(t1, t2):
    k,m,n = t1.shape
    _,l,p = t2.shape
    assert n == l
    t1_ff = np.fft.fft(t1, axis=0)
    t2_ff = np.fft.fft(t2, axis=0)
    t_r = np.zeros((k,m,p), dtype= complex)
    index = k+1//2
    for i in range(index):
        t_r[i,:,:] = t1_ff[i,:,:] @ t2_ff[i,:,:]
    for i in range(index, k):
        aux = k-i 
        t_r[i,:,:] = t_r[aux, :,:].conj()
    t_r = np.fft.ifft(t_r, axis=0)
    return t_r
"""
    The function `tensorSVD` performs Singular Value Decomposition (SVD) on a 3D tensor using Fast
    Fourier Transform (FFT) for efficient computation.
    
    :param tensor: It looks like the code you provided is a function called `tensorSVD` that performs
    Singular Value Decomposition (SVD) on a tensor. However, you have not provided the actual tensor
    data as input to the function
    :return: The function `tensorSVD` returns three arrays: `u_tensor`, `s_tensor`, and `v_tensor`.
    These arrays represent the left singular vectors, singular values, and right singular vectors
    respectively, obtained from performing Singular Value Decomposition (SVD) on the input tensor.
    """
def tensorSVD(tensor):
    tensor_a_aux = np.fft.fft(tensor, axis=0) # iterates the fast fourier transform in front slice
    k,m,n = tensor.shape
    index =((k + 1)//2) #aks for the midd value
    u_tensor =np.zeros((k,m,m), dtype=complex)
    s_tensor = np.zeros((k,m,n), dtype=complex)
    v_tensor =np.zeros((k,n,n), dtype=complex)
    tensor_a_aux[0,:,:] = tensor_a_aux[0,:,:].real
    for i in range(index): #iterates from the start until the mid matrix
        u_i, s_i, v_i = np.linalg.svd(tensor_a_aux[i,:,:],full_matrices=True) #Aks for the SVD of the front slice of the tesor
        u_tensor[i,:,:] = u_i
        s_matrix = np.zeros((m,n), dtype=complex)
        min_dim = min(m,n)
        s_matrix[:min_dim,:min_dim] = np.diag(s_i)
        s_tensor[i,:,:]= s_matrix #it most change for no square mmatrix
        v_tensor[i,:,:] = v_i.T
    for i in range(index,k): #iterates from the middle until the end of the tensor
        aux = k-i
        u_tensor[i,:,:] = np.conjugate(u_tensor[aux,:,:])
        s_tensor[i,:,:] = s_tensor[aux, :, :]
        v_tensor[i,:,:] = np.conjugate(v_tensor[aux,:,:])
    u_tensor = np.fft.ifft(u_tensor,axis=0) #Return the tensor to the continous domain
    v_tensor = np.fft.ifft(v_tensor,axis=0)
    s_tensor = np.fft.ifft(s_tensor,axis=0)
    v_tensor = np.transpose(v_tensor, (0,2,1)) #Transpose the tensor
    return u_tensor, s_tensor, v_tensor
   

#Tensor T
a = np.array([[1,2,3],[4,5,4,],[7,8,10]])
b = np.array([[10,20,8],[40,50,9],[70,80,1]])
c = np.array([[100,200,45],[400,500,45],[700,800,45]])
d = np.array([[100,200,45],[400,500,45],[700,800,45]])
tensor = np.array([a,b,c])
u_tensor, s_tensor, v_tensor =tensorSVD(tensor)
print(u_tensor.shape)
print(s_tensor.shape)
print(v_tensor.shape)
r = t_product(u_tensor, s_tensor)
r = t_product(r,v_tensor)
print("Tensor\n", r.real)
