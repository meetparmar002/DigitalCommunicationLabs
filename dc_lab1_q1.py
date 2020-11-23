import numpy as np
import scipy
import matplotlib.pyplot as plot


def mysinplot(f, fs, n):
    Ts = 1 / fs
    T = 1/f
    t = np.arange(0, T, Ts)
    sig = np.sin(2*np.pi*f*t)
    plot.plot(t, sig)


f = float(input('Enter the frequency: '))
fs = float(input('Enter the sampling frequency: '))
n = int(input("Enter the cycle: "))
mysinplot(f, fs, n)
