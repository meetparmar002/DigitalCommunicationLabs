import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n = 10  # length of the bit sequence
N = 1000  # number of sample functions/experiments
v_pos = 2  # V+
v_neg = -1*2  # V-


def func(T, Ts, N=1000, n=10):  # Part (b)
    mx = [0]*n
    Xn = []
    c = [v_neg, v_pos]
    for _ in range(10):
        for _ in range(N):
            bitSeq = []
            for i in range(n):
                bit = np.random.choice(c)
                bitSeq.append(bit)
            Xn.append(bitSeq)
        mx = np.add(mx, np.mean(Xn, axis=0))
    print('mx = ')
    print(mx/10)

    all_means = []

    # these loops are goint to find auto-correlation between all the pairs
    for i in range(n):
        for j in range(n):
            n1 = i
            n2 = j
            df = pd.DataFrame(Xn)
            # Rx(n1,n2)
            mx1 = np.mean(np.multiply(Xn[:N-1][n1], Xn[:N-1][n2]))
            # Rx(n1-n2,0)
            mx2 = np.mean(np.multiply(Xn[:N-1][np.abs(n1-n2)], Xn[:N-1][0]))
            all_means.append([mx1, mx2, mx1-mx2])
    print('   Rx(n1,n2),Rx(n1-n2,0),Rx(n1,n2)-Rx(n1-n2,0)')
    print(pd.DataFrame(all_means))
    print('As we can see that mx is approximatally zero(constant) and Rx(n1,n2) = Rx(n1-n2,0) (approximatly) we can say that this Stochastic Process X(n) is WSS.')

    generateDelayedNRZL(T, Ts, Xn)  # Part (c)


def generatePulse(T, Ts, Xn, t):
    bitSeq = []
    if t == 0:
        bitSeq = Xn[0]
    else:
        bitSeq = Xn[1]
    # print(bitSeq)
    for i in range(n):
        t = np.arange(i*T, (i+1)*T, Ts)
        if bitSeq[i] == v_pos:
            pulse_one = [v_pos]*int(T/Ts)
            plt.stem(t, pulse_one, 'g')
        else:
            pulse_zero = [v_neg]*int(T/Ts)
            plt.stem(t, pulse_zero, 'g')
    plt.xlabel('time(in seconds) --> ')
    plt.ylabel('Voltage(in volts) --> ')
    plt.title('A non-delayed NRZ-L signal')
    plt.show()


def generateDelayedNRZL(T, Ts, Xn):
    generatePulse(T, Ts, Xn, 0)
    d = np.random.random_integers(0, T+1, 1)
    print("Delay is %d" % d[0])
    if d[0] != 0:
        t = np.arange(0, d[0], Ts)
        plt.stem(t, np.zeros(len(t)), 'r')

    for i in range(n):
        t = np.arange(i*T+d[0], (i+1)*T+d[0], Ts)
        if Xn[0][i] == 2:
            pulse_one = [v_pos]*int(T/Ts)
            plt.stem(t, pulse_one, 'r')
        else:
            pulse_zero = [v_neg]*int(T/Ts)
            plt.stem(t, pulse_zero, 'r')
    plt.xlabel('time(in seconds) --> ')
    plt.ylabel('Voltage(in volts) --> ')
    plt.title('Delayed NRZ-L signal with delay = %d' % d[0])
    plt.show()

    generatePulse(T, Ts, Xn, 1)
    if d[0] != 0:
        t2 = np.arange(0, d[0], Ts)
        plt.stem(t2, np.zeros(len(t2)), 'r')
    for i in range(n):
        t = np.arange(i*T+d[0], (i+1)*T+d[0], Ts)
        if Xn[1][i] == 2:
            pulse_one = [v_pos]*int(T/Ts)
            plt.stem(t, pulse_one, 'r')
        else:
            pulse_zero = [v_neg]*int(T/Ts)
            plt.stem(t, pulse_zero, 'r')
    plt.xlabel('time(in seconds) --> ')
    plt.ylabel('Voltage(in volts) --> ')
    plt.title('Delayed NRZ-L signal(X1(n)) with delay = %d' % d[0])
    plt.show()

    calculatePSD(T, Ts, Xn)


def calculatePSD(T, Ts, Xn):
    ft_xn = np.fft.fft(Xn[0])
    ft_xn_conj = np.conjugate(ft_xn)

    mod_ft_xn = np.multiply(ft_xn, ft_xn_conj)
    Rx0 = np.mean(np.multiply(Xn[0], Xn[0]))

    PSD = (1 / T) * Ts * mod_ft_xn * Rx0

    plt.plot(PSD)
    # f = np.arange(-1 * 1 / T, 1 / T, 1 / Ts)
    # plt.plot(f, (v_pos ** 2) * T * np.sinc(f * T) * np.sinc(f*T))
    plt.show()


if __name__ == "__main__":
    T = float(input('Enter Width of the Pulse(in Seconds): '))
    Ts = float(input('Enter Sampling Interval(in Seconds): '))
    # generatePulse(T, Ts)
    func(T, Ts)
