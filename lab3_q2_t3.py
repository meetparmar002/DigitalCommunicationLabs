from math import sqrt
from numpy.core.umath import conjugate, multiply
from matplotlib.pyplot import *
from numpy import *
import math
# covariance matrix
sigma = matrix([[2.3, 0, 0, 0],
                [0, 1.5, 0, 0],
                [0, 0, 1.7, 0],
                [0, 0,   0, 2]
                ])
# mean vector
mu = array([2, 3, 8, 10])

# input
x = array([2.1, 3.5, 8, 9.5])


def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0 / (math.pow((2*pi), float(size)/2)
                            * math.pow(det, 1.0/2))
        x_mu = matrix(x - mu)
        inv = sigma.I
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")


print (norm_pdf_multivariate(x, mu, sigma))


def myawgn(psd, B, Fs, length):
    p = 2 * B * psd
    x = np.linspace(-1 * B, B, length)  # creating x-axis of freaquency
    sigma = sqrt(p)

    # AWGN noise
    awgn = sigma * random.randn(length)

    # getting the Estimated PSD
    temp_awgn = sigma * random.randn(length)
    AWGN_FT = fft.fft(temp_awgn)
    AWGN_FT_conj = conjugate(AWGN_FT)
    psd = abs(mean(multiply(AWGN_FT, AWGN_FT_conj)))
    estimated_psd = psd / (Fs * length)
    print('Estimated PSD = ' + float(estimated_psd))

    # ploting AWGN
    p1 = plot(x, awgn)
    xlabel('Frequency(Hz)')
    ylabel('AWGN')
    show()

psd = float(input('Enter PSD(in W/Hz): '))
B = float(input('Enter BandWidth(Hz): '))
Fs = float(input('Enter Sample Freaquency: '))
N = int(input('Enter the length of AWGN sequence: '))
myawgn(psd, B, Fs, N)



data = np. random. normal(0, 1, 1000) generate random normal dataset.
_, bins, _ = plt. hist(data, 20, density=1, alpha=0.5) 
mu, sigma = scipy. stats. norm. fit(data)
best_fit_line = scipy. stats. norm. pdf(bins, mu, sigma)
plt. plot(bins, best_fit_line)