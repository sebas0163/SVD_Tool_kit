import numpy as np
from Compact_SVD import compact_svd
from Complete_svd import full_svd
"""
    The function `svd` computes either a full or compact singular value decomposition of a given matrix
    based on the `complete` parameter.
    
    :param matrix: The `matrix` parameter in the `svd` function is typically a 2D array or matrix for
    which you want to compute the Singular Value Decomposition (SVD). The SVD is a factorization of a
    matrix into three matrices: U, Σ, and V^T
    :param complete: The `complete` parameter in the `svd` function is a boolean flag that determines
    whether to perform a full Singular Value Decomposition (SVD) or a compact SVD on the input matrix.
    If `complete` is `True`, a full SVD will be performed, while if it
    :return: The function `svd` returns the matrices `u`, `s`, and `v` which are the components of the
    Singular Value Decomposition (SVD) of the input matrix.
    """
def svd(matrix, complete):
    if complete:
        u,s,v = full_svd(matrix)
    else:
        u,s,v = compact_svd(matrix)
    return u,s,v