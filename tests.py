from T_SVD import T_SVD
from Q_SVD import Q_SVD
from SVD import svd
from HO_SVD import high_Order_SVD
from GSVD import GSVD
import numpy as np
import time as tm
import matplotlib.pyplot as plt

def err(t_org, t_const):
    num = abs(np.linalg.norm(t_org-t_const))
    denom = np.linalg.norm(t_org)
    if denom ==0:
        denom = 0.0000001
    return (num/denom)*100

def com_time__compact_SVD(m,n): #This algorithm compares the numpy compact svd and the experimental compact svd
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
GSVD_test(100,100)
