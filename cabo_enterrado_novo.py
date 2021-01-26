# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 20:12:14 2021

@author: luan9
"""

from numpy import exp, abs, angle, conj
import numpy as np
# fucoes que representam as funcoes de bessel
from scipy.constants import mu_0, epsilon_0
from scipy.special import k1, k0, i1, i0, yn
import scipy
from scipy.linalg import sqrtm
from math import log


class CaboEnterrado:

    def __init__(self, r1, r2, r3, r4, r5, r6, h, f, rhoa, rhoc, rhob, sigma, epsilon1, epsilon2, epsilon3, mu1, mu2):
        self.f = f
        self.omega = 2*np.pi*self.f
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4
        self.r5 = r5
        self.r6 = r6
        self.h = h
        self.rhoc = rhoc
        self.rhob = rhob
        self.rhoa = rhoa
        self.sigma = sigma
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.epsilon3 = epsilon3
        self.mu = mu_0
        self.mu1 = mu1
        self.mu2 = mu2
        self.delta = self.r3 - self.r2
        self.etac = np.sqrt((1j*self.omega*self.mu*self.mu1)/self.rhoc)
        self.etab = np.sqrt((1j*self.omega*self.mu*self.mu2)/self.rhob)
        self.etaA = np.sqrt((1j*self.omega*self.mu*self.mu2)/self.rhoa)
        self.delta = self.r3 - self.r2
        self.sum = self.r2 + self.r3
        self.delta1 = self.r5 - self.r4
        self.sum1 = self.r4 + self.r5

    def cZYcbi(self):
        # metodo para construir as matrizes de impedancia e admitancia para o cabo SC com blindagem, armadura e isolacao
        D = i1(abs(self.etab*self.r3))*k1(abs(self.etab*self.r2)) - \
            i1(abs(self.etab*self.r2))*k1(abs(self.etab*self.r3))

        Da = i1(abs(self.etaA*self.r5))*k1(abs(self.etaA*self.r4)) - \
            i1(abs(self.etaA*self.r4))*k1(abs(self.etaA*self.r5))

        Dal = i0(abs(self.etaA*self.r5))*k1(abs(self.etaA*self.r4)) + \
            i1(abs(self.etaA*self.r4))*k0(abs(self.etaA*self.r5))

        z1 = (self.rhoc*self.etac/(2*np.pi*self.r1)) * \
            (i0(abs(self.etac*self.r1))/i1(abs(self.etac*self.r1)))

        z2 = ((1j*self.omega*self.mu)/2*np.pi)*np.log(self.r2/self.r1)

        z3 = (self.rhob*self.etab/(2*np.pi*self.r2))*(i0(abs(self.etab*self.r2))*k1(abs(self.etab*self.r3)) +
                                                      k0(abs(self.etab*self.r2))*i1(abs(self.etab*self.r3)))/D

        z4 = (self.rhob/(2*np.pi*self.r2*self.r3))/D

        z5 = (self.rhob*self.etab/(2*np.pi*self.r3))*(i0(abs(self.etab*self.r3))*k1(abs(self.etab*self.r2)) +
                                                      k0(abs(self.etab*self.r3))*i1(abs(self.etab*self.r2)))/D

        z6 = ((1j*self.omega*self.mu)/2*np.pi)*np.log(self.r4/self.r3)

        z7 = self.etaA*self.rhoa / \
            (2*np.pi*self.r4)*np.arctanh(self.etab*self.delta1) - \
            self.rhoa/(2*np.pi*self.r4*self.sum)

        z8 = (1/(2*np.pi*self.r5*self.sigma)) * (1/Da)

        z9 = (self.etaA/(2*np.pi*self.r5*self.sigma)) * (Dal/Da)

        z10 = ((1j*self.omega*self.mu)/2*np.pi)*np.log(self.r6/self.r5)

        Zicc = z1 + z2 + z3 - 2*z4 + z5 + z6 + z7 - 2*z8 + z9 + z10

        Ziss = z5 + z6 + z7 - 2*z8 + z9 + z10

        Zics = z5 - z4 + z6 + z7 - 2*z8 + z9 + z10

        Ziaa = z9 + z10

        Zica = z9 - z8 + z10

        Zisa = z9 - z8 + z10

        Zi = np.array(
            [[Zicc, Zics, Zica], [Zics, Ziss, Zisa], [Zica, Zisa, Ziaa]])

        y1 = (1j*self.omega*2*np.pi*epsilon_0 *
              self.epsilon1)/log(self.r2/self.r1)

        y2 = (1j*self.omega*2*np.pi*epsilon_0 *
              self.epsilon2)/log(self.r4/self.r3)

        y3 = (1j*self.omega*2*np.pi*epsilon_0 *
              self.epsilon3)/log(self.r6/self.r5)

        Yi = np.array([[y1, -y1, 0], [-y1, y1+y2, -y2], [0, -y2, y2+y3]])

        array = [Zi, Yi]
        return(array)

    def cZsolo(self, h1, h2):
        # metodo para calcular a impedancia do solo
        eta = np.sqrt(1j*self.omega*self.mu *
                      (self.sigma+1j*self.omega*epsilon_0))

        Zsolo = (1j*self.omega*self.mu/(2*np.pi))*(k0(abs(eta*np.sqrt(self.r4 **
                                                                      2 + (h1-h2)**2)))+(((h1 + h2)**2 - self.r4**2)/(self.r4**2 + (h1+h2)**2))*yn(2, abs((eta*np.sqrt(self.r4**2 + (h1+h2)**2)) - 2*((exp((-h1-h2)*eta)*(1 + (h1+h2)*eta)))/((eta**2) * (self.r4**2 + (h1+h2)**2)))))

        return Zsolo

    def matrizes(self):
        array = self.cZYcbi()
        dim = (3, 3)
        Zint = np.block([[array[0], np.zeros(dim), np.zeros(dim)], [
                        np.zeros(dim), array[0], np.zeros(dim)], [np.zeros(dim), np.zeros(dim), array[0]]])

        Ycabo = np.block(
            [[array[1], np.zeros(dim), np.zeros(dim)], [np.zeros(dim), array[1], np.zeros(dim)], [np.zeros(dim), np.zeros(dim), array[1]]])

        matriz = [Zint, Ycabo]
        return matriz

    def matrizSolo(self):
        # metodo para montar a matriz de impedancias do solo
        dim = (3, 3)
        z0s = self.cZsolo(self.h, self.h) * np.ones(dim)
        z0m = self.cZsolo(self.h, self.h) * np.ones(dim)
        z0n = self.cZsolo(self.h, self.h) * np.ones(dim)

        Z0 = np.block([[z0s, z0m, z0n], [z0m, z0s, z0m], [z0n, z0m, z0s]])

        return Z0

    def modosPropagacao(self):
        # metodo para calcular as matrizes associadas aos modos de propagacao e a admitancia nodal
        matriz = self.matrizes()
        Z0 = self.matrizSolo()
        Zint = matriz[0]
        Y = matriz[1]
        M = (Zint+Z0)@Y
        autov, autovt = np.linalg.eig(M)
        
        d = np.complex128(np.sqrt(autov))
        
        Tv = autovt
        Ti = np.linalg.inv(Tv.transpose())
        one = np.eye(9)
        hm = np.exp(-d*5000.0)
        print(hm)
        #tentativa usando a funcao square
        #pot = np.square(hm, 2)
        #tentativa usando a funcao power
        pot = np.power(hm, 2, dtype = np.complex_)
        Am = (d*(1 + pot))/(1 - pot)
        Bm = (-2*d*hm)/(1 - pot)
        y11 = np.linalg.inv(Zint+Z0)@Tv@(Am*one)@Tv
        y12 = np.linalg.inv(Zint+Z0)@Tv@(Bm*one)@Ti
        #A = np.linalg.inv(Tv)@((Zint+Z0)@Y)@Tv 
        #gama1 = np.sqrt(A[0][0])
        
        Ynodal = [y11, y12]
        return Ynodal


#NumPy does not provide a dtype with more precision than C’s long double\; in particular, the 128-bit IEEE quad precision data type (FORTRAN’s REAL*16\) is not available.
