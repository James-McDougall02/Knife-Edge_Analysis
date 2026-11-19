import numpy as np
import pandas as pd
from scipy import special
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys
from pathlib import Path

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

# ---------------- multi-mode cavity fitting ----------------
# Replaces the single-Lorentzian-on-the-last-peak approach: all the cavity modes (fundamental,
# first order, ...) are fitted together as a sum of Lorentzians so overlapping modes are separated.

def multi_lorentzian(t, *params, n_modes=1, baseline_order=1, t_mid=0.0):
    """
    Sum of n_modes Lorentzians on a constant or linear baseline.

    params = [amp_0, width_0, centre_0, amp_1, width_1, centre_1, ..., offset, slope]
    (slope is omitted when baseline_order = 0)

    Same conventions as lorentzian(): amp = peak height above the baseline,
    width = HWHM (nm), centre = wavelength (nm). Mode 0 is the fundamental
    (longest wavelength), mode 1 the first-order mode, and so on.
    The baseline is offset + slope * (t - t_mid), so `offset` is the baseline
    value at t_mid (the middle of the fit window).
    """
    t = np.asarray(t, dtype=float)
    y = np.full_like(t, params[3 * n_modes])
    if baseline_order >= 1:
        y = y + params[3 * n_modes + 1] * (t - t_mid)
    for k in range(n_modes):
        amp, width, centre = params[3 * k: 3 * k + 3]
        y = y + amp * width**2 / ((t - centre)**2 + width**2)
    return y

def find_modes(lam, counts, n_modes=3, prominence=1000, distance=20, height=None):
    """
    Find cavity-mode peaks in a spectrum.

    Returns the indices of up to n_modes peaks, longest wavelength first:
    index 0 is the fundamental, 1 the first-order mode, 2 the next, ...
    If fewer than n_modes peaks pass the thresholds, fewer are returned.
    The defaults are the thresholds the old fundamental-finding code used.
    """
    if height is None:
        height = 2000 - np.min(counts)
    peaks, _ = find_peaks(counts, prominence=prominence, distance=distance, height=height)
    peaks = peaks[np.argsort(lam[peaks])[::-1]]
    return peaks[:n_modes]

def fit_modes(lam, counts, centres, widths, window, free_shape=True,
              width_bounds=(1.0, 10.0), max_shift=3.0, baseline_order=1):
    """
    Fit len(centres) Lorentzian modes plus a baseline to lam/counts inside
    window = (lo, hi) nm.

    centres, widths : starting wavelengths and HWHMs (nm), fundamental first
    free_shape=True : fit amplitude, width and centre of every mode. Centres
                      are bounded to +-max_shift of the start (never more than
                      0.4x the gap to the neighbouring mode, so modes cannot
                      swap places); widths are bounded to width_bounds.
    free_shape=False: hold centres and widths at the values given and fit only
                      the amplitudes and the baseline.

    Returns a dict with the windowed data, the fit, every parameter with its
    1-sigma error (NaN for held parameters), and a flag for any width/centre
    that ended up on a bound.
    """
    centres = np.asarray(centres, dtype=float)
    widths = np.asarray(widths, dtype=float)
    n = len(centres)
    m = (lam >= window[0]) & (lam <= window[1])
    t, y = lam[m], counts[m].astype(float)
    t_mid = 0.5 * (window[0] + window[1])
    ptp = max(y.max() - y.min(), 1.0)

    gap = np.min(np.diff(np.sort(centres))) if n > 1 else np.inf
    shift = min(max_shift, 0.4 * gap)

    p0, lb, ub = [], [], []
    for c, w in zip(centres, widths):
        h = y[np.argmin(np.abs(t - c))] - y.min()
        p0 += [min(max(0.7 * h, 0.0), 1.5 * ptp), w, c]
        lb += [0.0, width_bounds[0], c - shift]
        ub += [1.5 * ptp, width_bounds[1], c + shift]
    p0 += [y.min()]
    lb += [y.min() - ptp]  # baseline at t_mid; allowed slightly below the data minimum
    ub += [y.max()]
    if baseline_order >= 1:
        p0 += [0.0]
        lb += [-np.inf]
        ub += [np.inf]
    p0, lb, ub = np.array(p0), np.array(lb), np.array(ub)
    p0 = np.clip(p0, lb, ub)  # starting point must sit inside the bounds

    free = np.ones(len(p0), dtype=bool)
    if not free_shape:
        for k in range(n):
            free[3 * k + 1] = free[3 * k + 2] = False

    def f(t_, *q):
        p = p0.copy()
        p[free] = q
        return multi_lorentzian(t_, *p, n_modes=n, baseline_order=baseline_order, t_mid=t_mid)

    q_opt, q_cov = curve_fit(f, t, y, p0=p0[free], bounds=(lb[free], ub[free]), max_nfev=20000)
    popt = p0.copy()
    popt[free] = q_opt
    perr = np.full(len(p0), np.nan)
    perr[free] = np.sqrt(np.diag(q_cov))

    at_bound = False
    if free_shape:
        for k in range(n):
            for j, tol in ((1, 0.01), (2, 0.01)):  # width, centre (nm)
                i = 3 * k + j
                if popt[i] <= lb[i] + tol or popt[i] >= ub[i] - tol:
                    at_bound = True

    modes = []
    for k in range(n):
        a, w, c = popt[3 * k: 3 * k + 3]
        ea, ew, ec = perr[3 * k: 3 * k + 3]
        modes.append({'amplitude': a, 'amplitude_err': ea,
                      'width': w, 'width_err': ew,
                      'wavelength': c, 'wavelength_err': ec})
    return {
        'mask': m, 't': t, 'y': y, 'y_fit': f(t, *q_opt), 't_mid': t_mid,
        'n_modes': n, 'modes': modes, 'baseline_order': baseline_order,
        'offset': popt[3 * n], 'offset_err': perr[3 * n],
        'slope': popt[3 * n + 1] if baseline_order >= 1 else 0.0,
        'slope_err': perr[3 * n + 1] if baseline_order >= 1 else np.nan,
        'at_bound': at_bound,
    }

def plot_mode_fit(ax, lam, counts, fit, title=None):
    """Data, total fit and the individual Lorentzian modes (each drawn on the baseline)."""
    t = fit['t']
    baseline = fit['offset'] + fit['slope'] * (t - fit['t_mid'])
    ax.plot(lam, counts, color='0.6', label='Raw Data')
    ax.plot(t, fit['y'], 'k.', ms=3, label='Fitting window')
    ax.plot(t, fit['y_fit'], 'r', label=f"{fit['n_modes']}-mode fit")
    for k, md in enumerate(fit['modes']):
        ax.plot(t, lorentzian(t, md['amplitude'], md['width'], md['wavelength'], 0) + baseline,
                '--', lw=1, label=f"Mode {k}: {md['wavelength']:.2f} nm")
    ax.set_xlim(t.min() - 10, t.max() + 10)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Counts')
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)

def fit_to_row(fit):
    """
    Flatten a fit_modes() result into results-CSV columns.
    The original columns (Fitted Amplitude, Fitted Width, ...) hold the fundamental
    (mode 0), so everything downstream that reads them is unchanged. Modes 1, 2, ...
    are added as 'Mode k ...' columns after them.
    """
    f0 = fit['modes'][0]
    row = {
        'Fitted Amplitude': f0['amplitude'],
        'Amplitude Error': f0['amplitude_err'],
        'Fitted Wavelength': f0['wavelength'],
        'Wavelength Error': f0['wavelength_err'],
        'Fitted Width': f0['width'],
        'Width Error': f0['width_err'],
        'Fitted Offset': fit['offset'],
        'Offset Error': fit['offset_err'],
        'Fit Status': 'Fit successful' + (' (width/centre at bound)' if fit['at_bound'] else ''),
        'Fitted Slope': fit['slope'],
        'Slope Error': fit['slope_err'],
        'Modes Fitted': fit['n_modes'],
    }
    for k, md in enumerate(fit['modes'][1:], start=1):
        row.update({
            f'Mode {k} Amplitude': md['amplitude'],
            f'Mode {k} Amplitude Error': md['amplitude_err'],
            f'Mode {k} Wavelength': md['wavelength'],
            f'Mode {k} Wavelength Error': md['wavelength_err'],
            f'Mode {k} Width': md['width'],
            f'Mode {k} Width Error': md['width_err'],
        })
    return row

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
 