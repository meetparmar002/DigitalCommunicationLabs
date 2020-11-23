# Lab2 - Problem 1(c)
import numpy as np
import pandas as ps
import matplotlib.pyplot as plt

Xt = []
mx = 0
mx1 = 0
mx2 = 0
for _ in range(100):
    Xt = np.random.normal(loc=0, scale=1, size=(1000, 4))
    mx = mx + np.mean(Xt)
    n1, n2 = 1, 3
    x1, x2 = ps.DataFrame(Xt)[n1], ps.DataFrame(Xt)[n2]
    x0 = ps.DataFrame(Xt)[0]
    mx1 = mx1 + (x1 * x2).mean(axis=0)
    mx2 = mx2 + ((x1 - x2) * x0).mean(axis=0)

mx = mx / 100
mx1 = mx1 / 100
mx2 = mx2 / 100

# print(Xt[0])
print('The mean of mx = %.5f' % mx)
print('The mean of Rx(n1,n2)  = %.5f' % mx1)
print('The mean of Rx(n1-n2,0) = %.5f' % mx2)

x = ps.DataFrame(Xt)[0]
# print(x)
plt.hist(Xt[0:], bins='auto')
# plt.hist(x, bins='auto')
plt.title('Density function for X(n)')
plt.show()
