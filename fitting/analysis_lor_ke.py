import sys
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from scripts.lorentzian_fit import load_csv, lorentzian, fit_lorentzian # analyse spectra of pillars
from scripts.erf_fit import erf_func, fit_erf # analyse knife edge amplitude points

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent  # vscode
except NameError:
    #print('jup')
    PROJECT_ROOT = Path.cwd().parent # jup
sys.path.append(str(PROJECT_ROOT))

data_dir = PROJECT_ROOT.parent / "2608" / "260812" / "C6C-200-4A_FourierEdge_1e-4"
print("data location: ", data_dir)
### Find all CSV files in the data directory and count them
print(len(list(data_dir.iterdir())), "files found in data directory.")
files = [f for f in data_dir.glob("*.csv") if 'results' not in f.name.lower() ]
if files:
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
else:
    print("No matching CSV files found")


rows = [] # to hold results to put in csv file

lam, counts = load_csv(latest_file) # load data from .csv
peaks, props = find_peaks(counts, prominence=100, distance=60, height=1) # find peaks
long_lam_peak_idx = np.argmax(counts[peaks])
lam_fundamental = lam[peaks][long_lam_peak_idx]
counts_fundamental = counts[peaks][long_lam_peak_idx]
print("Fundamental wavelength: ", lam_fundamental)

lower_bounds = [0, 0, lam_fundamental - 10, 0] # amp, width, wavelength, offset
upper_bounds = [counts_fundamental*2, 100, lam_fundamental + 10, counts_fundamental] # amp, width, wavelength, offset

for f in files:
    row = {'File': f.name}

    try:
        lam, counts = load_csv(f)
        window = 15
        lam_window = (lam > lam_fundamental - window/2) & (lam < lam_fundamental + window/2)
        lam_windowed = lam[lam_window]
        counts_windowed = counts[lam_window]
        #print(f"File: {f.name}, \nFitting window: {lam_windowed.min()} - {lam_windowed.max()} nm")
        peaks, props = find_peaks(
            counts_windowed, 
            prominence=100, 
            distance=60, 
            height=1,
            )
        #print(f"File: {f.name}, Peaks found at wavelengths: {lam_windowed[peaks]}")
        if len(peaks) == 0:
            print("No peaks found in file: ", f.name)
            row.update({
                'Fitted Amplitude': np.nan,
                'Amplitude Error': np.nan,
                'Fitted Wavelength': np.nan,
                'Wavelength Error': np.nan,
                'Fitted Width': np.nan,
                'Width Error': np.nan,
                'Fitted Offset': np.nan,
                'Offset Error': np.nan,
                'Fit Status': 'No peak found'
            })
            rows.append(row)
            continue


        guesses = [{
            "amplitude": counts_windowed[peaks][0],
            "width": 5,
            "wavelength": lam_fundamental,
            "offset": np.min(counts_windowed)
        }]
        #print(f"File: {f.name}, Guessed parameters: ", guesses)

        popt, perr, y_fit = fit_lorentzian(
            lam_windowed,
            counts_windowed, 
            guesses,
            bounds=(lower_bounds, upper_bounds)
            )
        #print(f"File: {f.name}, Fitted parameters: ", popt)
        row.update({
            'Fitted Amplitude': popt[0],
            'Amplitude Error': perr[0],
            'Fitted Wavelength': popt[2],
            'Wavelength Error': perr[2],
            'Fitted Width': popt[1],
            'Width Error': perr[1],
            'Fitted Offset': popt[3],
            'Offset Error': perr[3],
            'Fit Status': 'Fit successful'
        })

        plot = 0 # input("y or n to plot? ")
        if plot == "y":
            fig, ax = plt.subplots()
            ax.plot(lam, counts, label='Raw Data')
            ax.plot(lam_windowed[peaks][0], counts_windowed[peaks][0], 'x', label='Peak for Fitting')
            ax.plot(lam_windowed, y_fit, label='Lorentzian Fit', color='red')
            ax.axvspan(lam_windowed.min(), lam_windowed.max(), color='yellow', alpha=0.3, label='Fitting Window')
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Counts')
            ax.set_title(f'Lorentzian Fit for {f.name}')
            ax.legend()

    except Exception as exc:
        print(f"Error processing file {f.name}")
        print(repr(exc))
        row.update({
            'Fitted Amplitude': np.nan,
            'Amplitude Error': np.nan,
            'Fitted Wavelength': np.nan,
            'Wavelength Error': np.nan,
            'Fitted Width': np.nan,
            'Width Error': np.nan,
            'Fitted Offset': np.nan,
            'Offset Error': np.nan,
        })

    rows.append(row)

results = pd.DataFrame(rows)
results.to_csv(data_dir / 'fit_results.csv', index=False)
print(results.head())
print(f"Saved {len(results)} rows to {data_dir / 'fit_results.csv'}")

"""
Error function analysis for amplitude over knife edge range 
""" 


df = pd.read_csv(data_dir / 'fit_results.csv')
fit_amp = df['Fitted Amplitude'].fillna(0)  # Fill NaN values with 0 for plotting
in_to_um = 2.54 * 1e4 # convert inches to micrometers
step_size_in = 1e-4 # step size in inches
step_to_um = in_to_um * step_size_in
ke_pos = step_to_um * np.arange(len(df))

erf_guess = [{
    "amplitude": np.max(fit_amp),
    "width": 5,
    "centre": np.argwhere(fit_amp == np.max(fit_amp))[0][0],  # Use the index of the maximum amplitude as the initial guess for the centre
    "offset": fit_amp.min()
}]
popt, perr, erf_fit = fit_erf(ke_pos, fit_amp, erf_guess)



x_fit = np.linspace(np.min(ke_pos), np.max(ke_pos), 1000)

fig, ax = plt.subplots()
ax.plot(ke_pos, fit_amp, marker='o', linestyle='-', color='blue')
ax.plot(x_fit, erf_func(x_fit, *popt), color='red', label=f'Error Function Fit \nWidth: {round(popt[1], 2)}$\\pm$ {round(step_to_um * perr[1], 2)} μm') #, \nAmplitude: {round(popt[0], 2)}, \nCentre: {round(step_to_um * popt[2], 2)} μm, \nOffset: {round(popt[3], 2)}')
#ax.set_xlim(50, 500)
#ax.set_ylim(0, 3000)
ax.set_xlabel('Knife Edge Position (μm)')
ax.set_ylabel('Fitted Amplitude (Hz)')
ax.set_title(f'Knife Edge Analysis: {data_dir.name}')
ax.legend()

save_figure = input("y or n to save figure? ")
if save_figure == "y":
    savefig = data_dir / f'{data_dir.name}_erf_fit.pdf'
    fig.savefig(savefig, dpi=300)
plt.show()