from numpy import exp, abs, angle, conj
import numpy as np
# fucoes que representam as funcoes de bessel
from scipy.constants import mu_0, epsilon_0
from scipy.special import k1, k0, i1, i0, yn
from math import log


class CaboEnterrado:

    def __init__(self, r1, r2, r3, r4, h, f, rhoc, rhob, sigma, epsilon1, epsilon2, mu1, mu2):
        self.f = f
        self.omega = 2*np.pi*self.f
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4
        self.h = h
        self.rhoc = rhoc
        self.rhob = rhob
        self.sigma = sigma
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.mu = mu_0
        self.mu1 = mu1
        self.mu2 = mu2
        self.delta = r3 - r2
        self.etac = np.sqrt((1j*self.omega*self.mu*self.mu1)/self.rhoc)
        self.etab = np.sqrt((1j*self.omega*self.mu*self.mu2)/self.rhob)

    def cZYcbi(self):
        D = i1(abs(self.etab*self.r3))*k1(abs(self.etab*self.r2)) - \
            i1(abs(self.etab*self.r2))*k1(abs(self.etab*self.r3))

        z1 = (self.rhoc*self.etac/(2*np.pi*self.r1)) * \
            (i0(abs(self.etac*self.r1))/i1(abs(self.etac*self.r1)))

        z2 = ((1j*self.omega*self.mu)/2)*np.log(self.r2/self.r1)

        z3 = (self.rhob*self.etab/(2*np.pi*self.r2))*(i0(abs(self.etab*self.r2))*k1(abs(self.etab*self.r3)) +
                                  k0(abs(self.etab*self.r2))*i1(abs(self.etab*self.r3)))/D

        z4 = (self.rhob/(2*np.pi*self.r2*self.r3))/D

        z5 = (self.rhob*self.etab/(2*np.pi*self.r3))*(i0(abs(self.etab*self.r3))*k1(abs(self.etab*self.r2)) +
                                  k0(abs(self.etab*self.r3))*i1(abs(self.etab*self.r2)))/D

        z6 = ((1j*self.omega*self.mu)/2)*np.log(self.r4/self.r3)

        Zi = np.array([[z1+z2+z3+z5+z6-2*z4, z5+z6-z4], [z5+z6-z4, z5+z6]])

        y1 = (1j*self.omega*2*np.pi*epsilon_0 *
              self.epsilon1)/log(self.r2/self.r1)

        y2 = (1j*self.omega*2*np.pi*epsilon_0 *
              self.epsilon2)/log(self.r4/self.r3)

        Y = np.array([[y1, -y1], [-y1, y1+y2]])

        array = [Zi, Y]
        return(array)

    def matrizes(self):
        array = self.cZYcbi()
        Zint = np.array([[array[0], 0, 0], [0, array[0], 0], [0, 0, array[0]]])

        Ycabo = np.array([[array[1], 0, 0], [0, array[1], 0], [0, 0, array[1]]])
        
        matriz = [Zint, Ycabo]
        return matriz

    def cZsolo(self, h1, h2):
        eta = np.sqrt(1j*self.omega*self.mu *
                      (self.sigma+1j*self.omega*epsilon_0))

        Zsolo = (1j*self.omega*self.mu/(2*np.pi))*(k0(abs(eta*np.sqrt(self.r4 **
                                                                  2 + (h1-h2)**2)))+(((h1 + h2)**2 - self.r4**2)/(self.r4**2 + (h1+h2)**2))*yn(2, abs((eta*np.sqrt(self.r4**2 + (h1+h2)**2)) - 2*((exp((-h1-h2)*eta)*(1 + (h1+h2)*eta)))/((eta**2) * (self.r4**2 + (h1+h2)**2)))))  

        return Zsolo

    def matrizSolo(self):
        z0s = self.cZsolo(self.h, self.h) * np.array([[1, 1], [1, 1]])
        z0m = self.cZsolo(self.h, self.h) * np.array([[1, 1], [1, 1]])
        z0n = self.cZsolo(self.h, self.h) * np.array([[1, 1], [1, 1]])

        Z0 = np.array([[z0s, z0m, z0n], [z0m, z0s, z0m], [z0n, z0m, z0s]])

        return Z0

    def modosPropagacao(self):
        array = self.cZYcbi()
        Z0 = self.matrizSolo()
        Y = self.matrizes()[1]
        autov, autovt = np.linalg.eig((array[0]+Z0)@Y)

        Tv = autovt
        Ti = np.linalg.inv(Tv.transpose())
        A = np.linalg.inv(Tv)@((array[0]+Z0)@Y)@Tv

        gama1 = np.sqrt(A[0][0])

        return autov, autovt

    def Ynodal(self):
        array = self.cZYcbi()
        Z0 = self.matrizSolo()
        matrizes = self.matrizes()
        Y = matrizes[1]
        Yc = np.linalg.inv(array[0]+Z0)@np.sqrt((array[0]+Z0)@Y)
        H = exp(np.sqrt(Y@(array[0]+Z0)))
        A = Yc@(1 + np.linalg.matrix_power(H, 2)
                )@np.linalg.inv(1 - np.linalg.matrix_power(H, 2))
        B = -2*Yc@H@np.linalg.inv(1 - np.linalg.matrix_power(H, 2))
        Y1 = [[A, B], [B, A]]
        return Y1

# besselK[2, ....] = yn(2, argument)
