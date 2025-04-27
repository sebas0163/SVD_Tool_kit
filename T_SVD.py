import numpy as np 
import time as tm

"""
    The function T_SVD performs Singular Value Decomposition (SVD) on a tensor in the Fourier domain and
    returns the decomposed components in the real domain.
    
    :param tensor: It looks like the code you provided is a function called `T_SVD` that performs
    Singular Value Decomposition (SVD) on a 3D tensor. The function takes a tensor as input and returns
    the decomposed components U, S, and V
    :return: The function `T_SVD` returns three arrays: `tensor_U`, `tensor_S`, and `tensor_V`. These
    arrays represent the matrices U, S, and V from the Singular Value Decomposition (SVD) of the input
    tensor in the Fourier domain.
    """
def T_SVD(tensor):
    k,m,n = tensor.shape #Takes the tensor's dimentions
    tensor_fou = np.fft.fft(tensor,axis=0) #Transforms the tensor to the fourier domain
    tensor_u_fou = np.zeros((k,m,m),dtype=complex)
    tensor_v_fou = np.zeros((k,n,n),dtype=complex)
    tensor_s_fou = np.zeros((k,m,n),dtype=complex)
    for i in range(k):
        u_,s_,v_ =np.linalg.svd(tensor_fou[i,:,:],full_matrices=True) #Calculates the SVD for every frontal slice of the tensor
        s_matrix = np.zeros((m,n), dtype=complex) #Transforms the eigenvector in a diagonal matrix 
        min_dim = min(m,n)
        s_matrix[:min_dim,:min_dim] = np.diag(s_)
        tensor_s_fou[i,:,:] = s_matrix #Assign the SVD to every frontal slice
        tensor_u_fou[i,:,:] = u_
        tensor_v_fou[i,:,:] = v_.conj().T #This is necesary because the algorithm gives the conjugate transpose
    tensor_U = np.fft.ifft(tensor_u_fou, axis=0).real #Come back to the real domain
    tensor_S = np.fft.ifft(tensor_s_fou, axis=0).real
    tensor_V = np.fft.ifft(tensor_v_fou, axis=0).real
    return tensor_U,tensor_S,tensor_V
"""
    The function `transpose_Tensor` transposes and conjugates the frontal slices of a given tensor.
    
    :param tensor_: The function `transpose_Tensor` takes a 3D tensor as input and transposes each
    frontal slice of the tensor by first conjugating it and then taking the transpose
    :return: The function `transpose_Tensor` returns a transposed version of the input tensor with some
    specific operations applied to each frontal slice of the tensor.
    """
def transpose_Tensor(tensor_):
    k = tensor_.shape[0]
    cop = tensor_.copy()
    tensor_[0,:,:] = tensor_[0,:,:].conj().T #Conjugate and transpose the first frontal slice of the tensor
    j = k -1
    for i in range(1,k):#Conjugate and transpose the frontal slice of the tensor starting in the slice two and changing the order
        tensor_[i,:,:] = cop[j,:,:].conj().T
        j-=1
    return tensor_
"""
    The function `lineartransform` applies a linear transformation to a given input `X` using either a
    function or matrix multiplication on the 3rd axis based on the provided transformation dictionary.
    
    :param X: The `X` parameter in the `lineartransform` function represents the input data that you
    want to transform using the specified transformation. It could be a numpy array or a tensor that you
    want to apply the transformation on
    :param transform: The `transform` parameter is expected to be a dictionary containing a key `'L'`
    which maps to either a callable function or a 3D array (matrix) that will be used for linear
    transformation. The function `lineartransform` checks if the value corresponding to the key `'L'`
    :return: The function `lineartransform` returns the result of applying either a callable function or
    matrix multiplication on the 3rd axis of the input `X` based on the provided `transform` dictionary.
    """
def lineartransform(X, transform):
    if callable(transform['L']):
        return transform['L'](X)
    else:
        # Apply matrix multiplication on the 3rd axis
        return np.tensordot(X, transform['L'], axes=(0, 2))
"""
    The function inverselineartransform applies the inverse of a linear transformation to input data X
    using the provided transformation dictionary.
    
    :param X: The `X` parameter in the `inverselineartransform` function represents the input data that
    you want to transform using the specified transformation. It could be a matrix, tensor, or any other
    suitable data structure depending on the context of your transformation
    :param transform: It seems like the definition of the `inverselineartransform` function is
    incomplete. You mentioned the `transform` parameter but did not provide its full details. Could you
    please provide more information about the `transform` parameter so that I can assist you better?
    :return: The function `inverselineartransform` is returning the result of applying the inverse
    linear transformation to the input `X`. If the inverse linear transformation is a callable function,
    it directly applies the function to `X`. Otherwise, it performs a tensor dot product between `X` and
    the inverse linear transformation matrix specified in the `transform` dictionary.
    """
def inverselineartransform(X, transform):
    if callable(transform['inverseL']):
        return transform['inverseL'](X)
    else:
        return np.tensordot(X, transform['inverseL'], axes=(0, 2))
"""
The function `t_product` performs tensor multiplication with optional linear transformations.
    
:param A: The function `t_product` takes three parameters `A`, `B`, and `transform`. The parameter
`A` is a 3-dimensional numpy array representing a tensor with shape `(n3, n1, n2)`
:param B: The parameter `B` in the `t_product` function is expected to be a 3-dimensional numpy
array representing a tensor. The shape of the tensor is used to perform calculations within the
function
:param transform: The `transform` parameter in the `t_product` function is a dictionary that
contains three keys: 'L', 'inverseL', and 'l'
:return: The function `t_product` returns the result of the tensor product operation between tensors
A and B, with optional transformation functions applied. The result is a complex-valued tensor C
with shape (n3, n1, m2). If both A and B are real-valued tensors, the function returns the real part
of the resulting tensor C."""
def t_product(A, B, transform=None):
    n3, n1, n2 = A.shape
    _, _, m2 = B.shape
    if B.shape[0] != n3 or B.shape[1] != n2:
        raise ValueError("Inner tensor dimensions must agree.")
    if transform is None:
        transform = {
            'L': lambda x: np.fft.fft(x, axis=0),
            'inverseL': lambda x: np.fft.ifft(x, axis=0),
            'l': n3
        }
    C = np.zeros((n3, n1, m2), dtype=complex)
    if transform['L'] == np.fft.fft or (callable(transform['L']) and transform['L'].__name__ == '<lambda>'):
        A_hat = transform['L'](A)
        B_hat = transform['L'](B)
        halfn3 = int(np.ceil((n3 + 1) / 2))
        for i in range(halfn3):
            C[i] = A_hat[i] @ B_hat[i]
        for i in range(halfn3, n3):
            C[i] = np.conj(C[n3 - i])
        C = transform['inverseL'](C)
    else:
        A_hat = lineartransform(A, transform)
        B_hat = lineartransform(B, transform)
        for i in range(n3):
            C[i] = A_hat[i] @ B_hat[i]
        C = inverselineartransform(C, transform)
    return C.real if np.isrealobj(A) and np.isrealobj(B) else C
