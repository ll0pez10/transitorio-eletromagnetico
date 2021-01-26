from cabo_enterrado_novo import CaboEnterrado
from scipy.fft import fft, fftfreq, ifft
import numpy as np
from scipy import signal

# dados do cabo
e1 = 3.31
e2 = 2.3
e3 = 10
rhoc = 1.75439*10**(-8)
rhob = 2.08333*10**(-7)
rhoa = 1/(0.798*10**(7))
sigma = 10**(-3)
hc = 1.2
r1 = 14.5*10**(-3)
r2 = 22.2*10**(-3)
r3 = 24.7*10**(-3)
r4 = 28.85*10**(-3)
r5 = 34.45*10**(-3)
r6 = 36*10**(-3)
x1 = 0
x3 = 0.3
x5 = 0.6
mu1 = 1
mu2 = 1

#valores associados a FFT, resolucao e intervalo de amostragem
N = 1024
T = 1.0/800.0

#FFT
x = np.linspace(-1*N*T, N*T, N, endpoint=False)
filtro = signal.blackman(N)
step = np.heaviside(x, 0.5)
step_fourier = fft(step*filtro)

freq = np.logspace(0, 5, 200)
nf = len(freq)

# laco de frquencias
for frequencia in freq:
    cabo = CaboEnterrado(r1, r2, r3, r4, r5, r6, hc, frequencia, rhoa, rhoc,
                         rhob, sigma, e1, e2, e3, mu1, mu2)
    cabo.cZYcbi()
    cabo.cZsolo(hc,hc)
    cabo.matrizes()
    cabo.matrizSolo()
    cabo.modosPropagacao()
    np.seterr('raise')
