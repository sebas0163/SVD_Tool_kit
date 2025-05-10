import numpy as np
from SVD_Tool_Kit.Compact_SVD import compact_svd

a = np.array([[1,2],[3,4]])
u,s,v = compact_svd(a)

print(u@s@v.T)