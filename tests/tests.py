import numpy as np
import time as tm
import quaternion
import matplotlib.pyplot as plt
from SVD_Tool_Kit import compact_svd,GSVD,T_SVD,high_Order_SVD,join_SVD,Q_SVD,t_product,transpose_Tensor,frob_for_quaternions,dot_product_quat,svd
def err_quat(Q_org, Q_const):
    num = abs(frob_for_quaternions(Q_org-Q_const))
    denom = frob_for_quaternions(Q_org)
    if denom ==quaternion.quaternion(0,0,0,0):
        denom = quaternion.quaternion(1,0,0,0)
    result=(num/denom)*100
    mag = (result.w**2 + result.x **2 + result.y **2+ result.z**2)**(1/2)
    return mag
def err(t_org, t_const):
    num = abs(np.linalg.norm(t_org-t_const))
    denom = np.linalg.norm(t_org)
    if denom ==0:
        denom = 0.0000001
    return (num/denom)*100
def com_time__compact_SVD(m,n):
    labels=[]
    times_exp =[]
    improve = []
    times_numpy =[]
    for i in range(10):
        mat = np.random.rand(m,n)
        labels.append(str(m)+"x"+str(n))
        m +=100
        n +=100
        start = tm.perf_counter()    
        svd(mat, complete=False)
        end = tm.perf_counter()
        tm_exp = end-start
        times_exp.append(tm_exp*1000)
        start = tm.perf_counter()
        np.linalg.svd(mat)
        end = tm.perf_counter()
        tm_nump = end-start
        times_numpy.append(tm_nump*1000)
        imp = 100*(tm_nump-tm_exp)/tm_nump
        if imp < 0:
            imp = 0
        improve.append(imp)
    plt.figure(figsize=(10,9))
    plt.plot(labels, times_exp, label='Experimental Time',marker="o")
    plt.plot(labels, times_numpy, label='Numpy Time',marker="s")
    plt.xlabel("Dimensions")
    plt.ylabel("Execution time (ms)")
    plt.legend()
    plt.show()
    plt.figure(figsize=(10,9))
    plt.bar(labels, improve, width=0.3)
    plt.xlabel("Dimensions")
    plt.ylabel("Algorithm improvement")
    plt.show()
def compact_err(m,n):
    errors =[]
    labels =[]
    for i in range(10):
        mat = np.random.rand(m,n)
        labels.append(str(m)+"x"+str(n))
        m +=100
        n +=100
        u,s,v = svd(mat, complete=False)
        reconst = u @ s @ v.T
        errors.append(err(mat,reconst))
    plt.figure(figsize=(10,9))
    plt.bar(labels, errors, width=0.5)
    plt.xlabel("Dimensions")
    plt.ylabel("Error (%)")
    plt.title("Reconstruction Error")
    plt.show()
def GSVD_test(m,n):
    labels =[]
    times = []
    errors_A =[]
    errors_B =[]
    for i in range(10):
        mat_A = np.random.rand(m,n)
        mat_B = np.random.rand(m,n)
        labels.append(str(m)+"x"+str(n))
        m +=100
        n +=100
        start = tm.perf_counter()
        u_a,u_b,d_a,d_b,x = GSVD(mat_A,mat_B)
        end = tm.perf_counter()
        times.append((end-start)*1000)
        reconst_a = u_a @ d_a @ np.linalg.inv(x)
        reconst_b = u_b @ d_b @ np.linalg.inv(x)
        errors_A.append(err(mat_A,reconst_a))
        errors_B.append(err(mat_B,reconst_b))
    plt.figure(figsize=(10,9))
    plt.plot(labels, times, label='Experimental time',marker="o")
    plt.xlabel("Dimensions")
    plt.ylabel("Execution time (ms)")
    plt.title("Execution time for differents dimentions")
    plt.show()
    plt.figure(figsize=(10,9))
    plt.bar(labels, errors_A, width=0.5, label = "Matrix A")
    plt.bar(labels, errors_B, width=0.5, label = "Matrix B")
    plt.xlabel("Dimensions")
    plt.ylabel("Error (%)")
    plt.title("Reconstruction Error")
    plt.show()
def HO_test(N):
    matrix_list =[]
    times =[]
    labels =[]
    errors =[]
    for i in range(5):
        for r in range(N):
            a =np.random.rand(100,100)
            matrix_list.append(a)
        start = tm.perf_counter()
        m,l,o=high_Order_SVD(matrix_list=matrix_list)
        end = tm.perf_counter()
        times.append((end-start)*1000)
        labels.append(N)
        N+=1
        org =matrix_list[0]
        mat_exp=m[0]@l[0]@o.T
        error = err(org,mat_exp)
        errors.append(error)
    plt.figure(figsize=(10,9))
    plt.plot(labels, times, label='Experimental time',marker="o")
    plt.xlabel("Dimensions")
    plt.ylabel("Execution time (ms)")
    plt.title("Execution time for N matrix of 100x100")
    plt.show()
    plt.figure(figsize=(10,9))
    plt.bar(labels, errors, width=0.3)
    plt.xlabel("Dimensions")
    plt.ylabel("Error (%)")
    plt.title("Reconstruction Error for N matrix of 100x100")
    plt.show()
def test_T_SVD(k,m,n):
    errors =[]
    times =[]
    labels =[]
    for i in range(10):
        tensor = np.random.rand(k,m,n)
        labels.append(str(k)+"x"+str(m)+"x"+str(n))
        m +=10
        n +=10
        k+=10
        start = tm.perf_counter()    
        u_tensor, s_tensor, v_tensor =T_SVD(tensor)
        v = transpose_Tensor(v_tensor)
        end = tm.perf_counter()
        r = t_product(u_tensor, s_tensor)
        r = t_product(r,v_tensor)
        errors.append(err(tensor,r))
        times.append((end-start)*1000)
    plt.figure(figsize=(10,9))
    plt.plot(labels, times, label='Experimental Time',marker="o")
    plt.xlabel("Dimensions")
    plt.ylabel("Execution time (ms)")
    plt.legend()
    plt.show()
    plt.figure(figsize=(10,9))
    plt.bar(labels, errors, width=0.3)
    plt.xlabel("Dimensions")
    plt.ylabel("Error (%)")
    plt.title("Reconstruction Error for N matrix of 100x100")
    plt.show()
def test_joint_svd(k,m,n):
    labels = []
    times =[]
    errors=[]
    values =[]
    for _ in range (10):
        tensor = np.random.rand(k,m,n)
        start = tm.perf_counter()
        _,_,_,err,_,value_of = join_SVD(tensor)
        end = tm.perf_counter()
        times.append(end-start)
        labels.append(str(k)+"x"+str(m)+"x"+str(n))
        errors.append(err)
        values.append(value_of)
        k +=1
        n +=1
        m +=1
    plt.figure(figsize=(10,9))
    plt.plot(labels, times, label='Experimental Time',marker="o")
    plt.xlabel("Dimensions")
    plt.ylabel("Execution time (ms)")
    plt.legend()
    plt.show()
    plt.figure(figsize=(10,9))
    plt.bar(labels, errors, width=0.3)
    plt.xlabel("Dimensions")
    plt.ylabel("Error (%)")
    plt.title("Reconstruction Error")
    plt.show()
    plt.figure(figsize=(10,9))
    plt.bar(labels, values, width=0.3)
    plt.xlabel("Dimensions")
    plt.ylabel("Minimized error")
    plt.title("Minimized Error between matrix")
    plt.show()
def test_Q_SVD(n):
    Q_mat =[]
    labels =[]
    times = []
    errors=[]
    for _ in range(10):
        Q_mat=[]
        for _ in range(n):
            row = []
            for _ in range(n):
                q = quaternion.quaternion(np.random.random(),np.random.random(),np.random.random(),np.random.random())
                row.append(q)
            Q_mat.append(row)
        Q_mat = np.array(Q_mat)
        start = tm.perf_counter()
        u,v,s=Q_SVD(Q_mat)
        end = tm.perf_counter()
        times.append((end-start)*1000)
        labels.append(str(n)+"x"+str(n))
        r = dot_product_quat(u,s)
        r = dot_product_quat(r, v.conj().T)
        error = err_quat(Q_mat,r)
        errors.append(error)
        n+=10
    plt.figure(figsize=(10,9))
    plt.plot(labels, times, label='Experimental Time',marker="o")
    plt.xlabel("Dimensions")
    plt.ylabel("Execution time (ms)")
    plt.legend()
    plt.show()
    plt.figure(figsize=(10,9))
    plt.bar(labels, errors, width=0.3)
    plt.xlabel("Dimensions")
    plt.ylabel("Error (%)")
    plt.title("Reconstruction Error for a Quaternion Matrix of differents dimensions")
    plt.show()