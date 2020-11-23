# Lab2 - Problem 3
# This is for General Matrix R and C
import numpy as np
import pandas as ps
import matplotlib.pyplot as plt

x = np.random.random((10, 1)) * 10

# for R
R = x * x.T  # autocorrelation matrix

print('Matrix R: \n\n')
print(ps.DataFrame(R))
if R.all() == R.T.all():
    print('\nMatrix R is symmetric...')
else:
    print('R is not SPD. Run the code again to the results...')
    exit()

# checking Symmatric property by Eigen Values(Eigen values must be +ve so as to be Symmatric Matrix)
print('\nEigen values for R: ')
print(ps.DataFrame(np.around(np.linalg.eigvals(R))))

flag = True
for _ in range(1000):  # repeating so many times so that we can get better results
    y = np.random.random((10, 1)) * 10
    yTRy = ((y.T * R) * y)[0][0]
    if yTRy < 0:
        print('R is Not SPD')
        flag = False
        break

if flag:
    print('\nR is Symmetric Positive Semi-definite.')


# for C
m = np.random.random((10, 1)) * 10  # mean

C = (x - m) * (x - m).T  # auto-covariance
print('\nMatrix C: \n')
print(ps.DataFrame(C))

if C.all() == C.T.all():
    print('Matric C is Symmatric...')
else:
    print('C is not SPD. Run the code again to the results...')
    exit()
# checking Symmatric property by Eigen Values(Eigen values must be +ve so as to be Symmatric Matrix)
print('\nEigen values for C: ')
print(ps.DataFrame(np.around(np.linalg.eigvals(C))))


flag = True
for _ in range(1000):
    y = np.random.random((10, 1)) * 10
    yTRy = ((y.T * R) * y)[0][0]
    if yTRy < 0:
        print('C is Not SPD')
        flag = False
        break

if flag:
    print('\nC is Symmetric Positive Semi-definite.')
