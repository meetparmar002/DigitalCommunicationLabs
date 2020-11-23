# Lab2 - Problem 1(b)
import numpy as np
import pandas as ps
import matplotlib.pyplot as plt


def checkWSS():

    # Random Process (in 2D-matrix form where each column represents a RV(Xi) at n=i)
    Xt = []
    # number of sample functions/realizations
    N = 1000
    # creating mx array with all zeroes to store the mean for each column
    mx = [0] * 8

    A = np.random.uniform(-5, 5.0000001, N)

    mx1 = 0
    mx2 = 0

    for _ in range(100):  # iterating the whole process 100 times (to get the avg mean)
        for i in range(N):  # to create N sample functions
            # a sample function
            xi = [A[i]*np.cos(0.2 * np.pi * n) for n in range(8)]
            Xt.append(xi)  # add it to the Xt
        df = ps.DataFrame(Xt)
        mx = np.add(mx, df.mean(axis=0))
    print('col  mx')
    print(mx/100)  # printing the averaged mean

    all_means = []
    auto_co_mean = 0
    # these loops are goint to find auto-correlation between all the pairs
    for i in range(8):
        for j in range(8):
            n1 = i
            n2 = j
            # Rx(n1,n2)
            mx1 = np.mean(A*np.cos(0.25 * np.pi * n1) *
                          A * np.cos(0.25 * np.pi * n2))
            # Rx(n1-n2,0)
            mx2 = np.mean(A*np.cos(0.25 * np.pi * (n1-n2)) * A)
            auto_co_mean = auto_co_mean+mx1-mx2
            all_means.append([mx1, mx2, mx1-mx2])
    print('   Rx(n1,n2),Rx(n1-n2,0),Rx(n1,n2)-Rx(n1-n2,0)')
    print(ps.DataFrame(all_means))

    # picking the RV for Histogram
    cols = [int(np.random.uniform(0, 8)) for _ in range(4)]
    for i in cols:
        xi = np.cos(A*0.25*np.pi*i)
        plt.figure(i)
        plt.hist(xi, bins='auto')
        plt.title('Estimated density function for X' + str(i))
        plt.show()


if __name__ == '__main__':
    checkWSS()
