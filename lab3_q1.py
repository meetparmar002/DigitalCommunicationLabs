from numpy import *
from matplotlib.pyplot import *
from numpy.core.umath import conjugate, multiply
from math import sqrt


def myawgn(psd, B, Fs, length):
    p = 2 * B * psd
    x = np.linspace(-1 * B, B, length)  # creating x-axis of freaquency
    sigma = sqrt(p)
    print('sigma {}'.format(sigma))

    # AWGN noise
    awgn = sigma * random.randn(length)

    # getting the Estimated PSD
    estimated_psd = 0.0
    for _ in range(300):
        temp_awgn = sigma * random.randn(length)
        AWGN_FT = fft.fft(temp_awgn)
        AWGN_FT_conj = conjugate(AWGN_FT)
        psd = abs(mean(multiply(AWGN_FT, AWGN_FT_conj)))
        estimated_psd = estimated_psd + psd / (Fs * length)
    print('Estimated PSD = %.4f' % float(estimated_psd/300))

    # ploting AWGN
    figure(1)
    p1 = plot(x, awgn, label='AWGN')
    xlabel('Frequency(Hz)')
    ylabel('AWGN')
    title('AWGN Sequence')
    legend(loc='upper right')
    show()

    # ploting the Estimated PSD
    figure(2)
    y = [estimated_psd/300] * length
    p2 = plot(x, y, label='Estimated PSD')
    xlabel('Frequency(Hz)')
    ylabel('Estimated PSD')
    title('Estimated PSD')
    legend(loc='upper right')
    show()


if __name__ == '__main__':
    psd = float(input('Enter PSD(W/Hz): '))
    B = float(input('Enter BandWidth(Hz): '))
    Fs = float(input('Enter Sampling Freaquency: '))
    length = int(input('Enter the desired length of AWGN sequence: '))
    myawgn(psd, B, Fs, length)
