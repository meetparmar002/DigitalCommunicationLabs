import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy import random
import numpy.matlib as mlib
import numpy.linalg as lin


nexp = 100


def myfun(T, n_bits, samp_t):
    n = int(np.floor(T/samp_t))
    N = int(n_bits*n)
    delay = np.random.randint(n, size=(1, nexp))
    delay_max = np.max(delay)
    delay_min = np.min(delay)

    X1 = np.zeros((nexp, N))
    X2 = np.zeros((nexp, N+delay_min))
    X3 = np.zeros((nexp, N+delay_max))

    for i in range(nexp):
        x_n = np.random.randint(2, size=(1, n_bits))
        x2 = np.transpose(mlib.repmat(x_n, n, 1))
        x = np.reshape(x2, N)
        X1[i, :] = x
        d = int(delay[0][i])
        X3[i, d:d+len(x)] = x

    X2 = X3[:, 0:(N+delay_min)]
    X1_mean = np.mean(X1, axis=0)
    X2_mean = np.mean(X2, axis=0)

    fig = plt.figure()
    plt.plot(X1_mean)
    plt.plot(X2_mean)
    plt.plot(X1[1])
    plt.plot(X2[1])

    Rx2 = np.zeros((X1.shape[1], X1.shape[1]))

    for i in range(X1.shape[1]):
        for j in range(X1.shape[1]):
            su = 0
            for k in range(X1.shape[0]):
                su = su+X1[k][i]*X1[k][j]
            Rx2[i][j] = su/nexp

    Rp = np.zeros((Rx2.shape[0]))
    for i in range(Rx2.shape[0]):
        m = np.diag(Rx2, i)
        Rp[i] = np.mean(m)

    Rxd2 = np.zeros((X2.shape[1], X2.shape[1]))
    for j in range(X2.shape[1]):
        sud = 0
        for k in range(X2.shape[0]):
            sud = sud+X2[k][i]*X2[k][j]
        Rxd2[i][j] = sud/nexp

    Rpd = np.zeros((Rxd2.shape[0]))
    for i in range(Rxd2.shape[0]):
        md = np.diag(Rxd2, i)
        Rpd[i] = np.mean(md)

    Rp2 = np.flip(Rp, 0)
    Rp1 = np.append(Rp2, Rp)
    Rpd2 = np.flip(Rpd, 0)
    Rpd1 = np.append(Rpd2, Rpd)

    Y = np.fft.fft(Rp1)/len(Rp1)
    freq = np.fft.fftfreq(len(Rp1))*(1/samp_t)

    Yd = np.fft.fft(Rpd1)/len(Rpd1)
    freqd = np.fft.fftfreq(len(Rpd1))*(1/samp_t)

    # print('Please look here = %d' % len(Rp1))
    fig = plt.figure()
    plt.plot(Rp1)

    fig = plt.figure()
    plt.plot(Rpd1)

    fig = plt.figure()
    plt.plot(freq, abs(Y))

    fig = plt.figure()
    plt.plot(freq, abs(Yd))

    tau = 1
    limit = int(len(X1[0]))-2*tau
    R2xx = np.zeros((limit))

    for k in range(limit):
        su = 0
        for i in range(nexp):
            su = su+X1[i][k]*X1[i][k+tau]

            R2xx[k] = su/nexp

    fig = plt.figure()
    plt.plot(R2xx)
    plt.plot(X2[1])

    #tau=0;
    limit = int(len(X2[0]))-2*tau
    R2xxd = np.zeros((limit))

    for k in range(limit):
        sud = 0
        for i in range(nexp):
            sud = sud+X1[i][k]*X1[i][k+tau]

            R2xxd[k] = sud/nexp

    fig = plt.figure()
    plt.plot(R2xxd)
    plt.plot(X2[0])
    return


T = float(input("Enter width of pulse: "))
n_bits = int(input("Enter total bits: "))
samp_t = float(input("Enter sampling time interval: "))
myfun(T, n_bits, samp_t)
