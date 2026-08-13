from scipy import special
from scipy.optimize import curve_fit
import numpy as np

def erf_func(x, amplitude, width, centre, y0):
    return amplitude * (1 + special.erf((x - centre) / (np.sqrt(2) * width))) / 2 + y0

def fit_erf(x, y, erf_guess):
    amplitude0 = erf_guess[0]["amplitude"]
    width0 = erf_guess[0]["width"]
    centre0 = erf_guess[0]["centre"]
    y0 = erf_guess[0]["offset"]

    p0 = [amplitude0, width0, centre0, y0]

    popt, pcov = curve_fit(erf_func, x, y, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    y_fit = erf_func(x, *popt)
    return popt, perr, y_fit