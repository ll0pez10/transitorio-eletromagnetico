def filtros(filtro, n):

    if filtro == "hanning":
        sigma = sin((np.pi*1j)/n)/((np.pi*1j)/n)

    elif filtro == "hamming":
        alpha = 0.53836
        sigma = alpha + (1-alpha)*cos((2*np.pi*1j)/n)

    elif filtro == "blackman":
        alpha = 0.5
        beta = 0.08
        sigma = alpha - beta + alpha * \
            cos((np.pi*1j)/n) + beta*cos((2*np.pi*1j)/n)

    elif filtro == "Von Hann":
        alpha = 0.5
        sigma = alpha*(1 + cos((2*np.pi*1j)/n))

    elif filtro == "exact blackman":
        sigma = 0.35875 - 0.48829*Cos(((2*np.pi)/n)*(j - 1)) + 0.14128*Cos(
            ((2*np.pi)/n)*(2j - 2)) - 0.01168*Cos(((2*np.pi)/n)*(3j - 3))

    else:
        sigma = alpha + (1-alpha)*cos((2*np.pi*1j)/n)

    return sigma
