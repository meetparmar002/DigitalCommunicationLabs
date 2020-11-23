import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from math import exp
from sklearn.datasets import make_spd_matrix
import seaborn as sbn


def mygauss(m, C, s, n):
    # Cholesky Decomposition. which gives us lower triangular matrix L such that C=L*L.T
    L = np.linalg.cholesky(C)

    Nora = np.random.randn(n, s)

    # This equation gives us multivariate normal distribution(i.e x = m + L*Nora)
    x = np.matmul(L, Nora) + m
    x_df = pd.DataFrame(x)
    print(x_df)
    
    if n == 1:
        plt.figure(1)
        _, bins, _ = plt.hist(x[0], 20, density=1, alpha=0.5) 
        fit_to_hist = sc.stats.norm.pdf(bins, float(m), float(C))
        plt.plot(bins,fit_to_hist)
        plt.title('Univariate Normal Distribution')
        plt.show()
    elif n==2:
        plt.figure(1)
        plt.plot(x[0],x[1],'g.','g-',label='Bivariate Normal Distribution')
        plt.title('Bivariate Normal Distribution')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        
        sbn.jointplot(x=x[0],y=x[1],kind='kde',scale=0.0001)
        plt.title('Contour Representation of Bivariate Normal Distribution')
        plt.show()
        


if __name__ == '__main__':
    n = int(input('Enter the #of Random Variable(s)(Enter 1 or 2 for Graphs): '))

    # mean matrix
    m = np.random.random((n, 1))*2
    # print(m)
    # print(len(m))

    # covariance matrix which is symmatric and positive semidefinite
    C = make_spd_matrix(n)
    # print(C)

    s = int(input('Enter #of sample(s)(Enter high #of sample for better estimation): '))
    mygauss(m, C, s, n)
