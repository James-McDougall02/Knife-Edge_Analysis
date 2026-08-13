import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def load_csv(path, time_col=0, counts_col=1, **kwargs):
    """
    Load a two-column CSV of (time, counts) data.

    path        : path to the CSV file
    time_col    : column index or name for the time axis
    counts_col  : column index or name for the counts axis
    kwargs      : passed through to pandas.read_csv
                  (e.g. skiprows=1 if there's a header line,
                  or delimiter='\\t' for tab-separated files)

    Returns (t, y) as numpy arrays.
    """
    df = pd.read_csv(path, **kwargs)
    t = df.iloc[:, time_col].to_numpy() if isinstance(time_col, int) else df[time_col].to_numpy()
    y = df.iloc[:, counts_col].to_numpy() if isinstance(counts_col, int) else df[counts_col].to_numpy()
    return t, y

def lorentzian(t, amplitude, width, wavelength, y0):
    """
    Single Lorentzian peak.
    amplitude : peak height scaling
    wavelength : Lorentzian centre
    width     : Lorentzian half-width at half-maximum (HWHM)
    y0        : baseline offset
    """
    return amplitude * (width**2 / ((t - wavelength)**2 + width**2)) + y0

def fit_lorentzian(t, y, initial_guesses, bounds=None):
    amplitude0 = initial_guesses[0]["amplitude"]
    width0 = initial_guesses[0]["width"]
    wavelength0 = initial_guesses[0]["wavelength"]
    y0 = initial_guesses[0]["offset"]
    
    
    p0 = [amplitude0, width0, wavelength0, y0]

    if bounds is not None:
        popt, pcov = curve_fit(lorentzian, t, y, p0=p0, bounds=bounds)
    else:
        popt, pcov = curve_fit(lorentzian, t, y, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    y_fit = lorentzian(t, *popt)
    return popt, perr, y_fit