# Lab2 - Problem 1(a)
import numpy as np
import pandas as ps
import matplotlib.pyplot as plt

# Random Process (in 2D-matrix form where each column represents a RV(Xi) at n=i)
Xt = []
# number of sample functions/realizations
N = 1000
# creating mx array with all zeroes to store the mean for each column
mx = [0] * 10
# theta belongs to [-pi,pi] uniformly. Size will give N random values for theta
theta = np.random.uniform(low=-1 * np.pi, high=np.pi, size=N)

mx1 = 0
mx2 = 0
for _ in range(100):  # iterating the whole process 10 times (to get the avg mean)
    for i in range(N):  # to create N sample functions
        # a sample function
        xi = [np.cos(0.2 * np.pi * n + theta[i]) for n in range(10)]
        Xt.append(xi)  # add it to the Xt
    df = ps.DataFrame(Xt)
    # summing each and every mean of respective columns of Xt
    mx = np.add(mx, df.mean(axis=0))
# print(ps.DataFrame(Xt))

print('col  mx')
print(mx / 100)  # printing the averaged mean

all_means = []
# these loops are goint to find auto-correlation between all the pairs
for i in range(10):
    for j in range(10):
        n1 = i
        n2 = j
        # Rx(n1,n2)
        mx1 = np.mean(np.cos(0.2 * np.pi * n1 + theta)
                      * np.cos(0.2 * np.pi * n2 + theta))
        # Rx(n1-n2,0)
        mx2 = np.mean(np.cos(0.2 * np.pi * (n1-n2) + theta)
                      * np.cos(theta))
        all_means.append([mx1, mx2, mx1-mx2])
print('   Rx(n1,n2),Rx(n1-n2,0),Rx(n1,n2)-Rx(n1-n2,0)')
print(ps.DataFrame(all_means))

# picking the RV
cols = [int(np.random.uniform(0, 10)) for _ in range(4)]
for i in cols:
    xi = np.cos(0.2*np.pi*i+theta)
    plt.figure(i)
    plt.hist(xi, bins='auto')
    plt.title('Estimated density function for X' + str(i))
    plt.show()
