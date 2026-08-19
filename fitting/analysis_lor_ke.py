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

# RawCounts, PowerNorm
folder_dir = PROJECT_ROOT.parent / "2608" / "260819" # Folder for measurements (one per day)
normalised = 0 # 0 for RawCounts, 1 for PowerNorm data folders
if normalised == 0:
    data_norm_yn = "RawCounts"
elif normalised == 1:
    data_norm_yn = "PowerNorm" 

print(f"Searching for {data_norm_yn} folders in ", str(folder_dir))
for data_folder in folder_dir.glob("*/" + data_norm_yn): # search for subfolders in folder_dir (daily folder) which have a PowerNorm subfolder
    if data_folder.is_dir(): # check its a folder (directory)
        print(f"Found {data_folder.name} folder in: ", data_folder.parent.name, data_folder.parent.name)
        files = [f for f in data_folder.glob("*.csv") if 'results' not in f.name.lower() ]
        if files:
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            print("Latest File: ", latest_file.name)
            print(len(files), "matching CSV files found in ", data_folder.parent.name, data_folder.name)
        else:
            print("No matching CSV files found in ", data_folder.name)

for data_folder in folder_dir.glob("*/" + data_norm_yn): # find all subfolders in the daily global folder 
    if not data_folder.is_dir(): # skip anything that isn't a folder
        print("Skipping non-directory: ", data_folder.name)
        continue


    print(f"Found {data_folder.name} folder in: ", data_folder.parent.name)
    
    files = [f for f in data_folder.glob("*.csv") if 'results' not in f.name.lower() ]
    #data_dir = folder
    #print("data dir: ", data_dir)
    if files:
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        print(len(files), "matching CSV files found in ", data_folder.parent.name,"/", data_folder.name)
    else:
        print("No matching CSV files found in ", data_folder.parent.name,"/", data_folder.name)

    rows = [] # to hold results to put in csv file

    lam, counts = load_csv(latest_file) # load data from .csv
    peaks, props = find_peaks(counts, prominence=100, distance=60, height=1) # find peaks
    long_lam_peak_idx = np.argmax(counts[peaks]) # find peak with largest index (longest wavelength)
    lam_fundamental = lam[peaks][long_lam_peak_idx] # get the wavelength of the fundamental peak
    counts_fundamental = counts[peaks][long_lam_peak_idx] # get the counts of the fundamental peak
    print("Fundamental wavelength for data folder ", data_folder.parent.name,"/", data_folder.name, ": ", lam_fundamental)

    lower_bounds = [0, 0, lam_fundamental - 10, 0] # amp, width, wavelength, offset
    upper_bounds = [counts_fundamental*2, 100, lam_fundamental + 10, 1 * counts_fundamental] # amp, width, wavelength, offset
    print("Lower bound amplitude: ", lower_bounds[0])
    print("Upper bound amplitude: ", upper_bounds[0])

    files_peakless = []
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
                #print("No peaks found in file: ", f.name, "in data folder: ", data_folder.name)
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
                files_peakless.append(f.name)
                continue

            #print(len(files_peakless), "files with no peaks found in ", data_folder.name)
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

            plot = 'n' # input("y or n to plot? ")
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
        #print(len(files_peakless), "files with no peaks found in ", data_folder.name)

    print(len(files_peakless), "files with no peaks found in ", data_folder.parent.name, data_folder.name) # how many files had no peaks in *Pillar measurement* / PowerNorm or RawCounts folder

    results = pd.DataFrame(rows)
    results.to_csv(data_folder / 'fit_results.csv', index=False) # save results to PowerNorm or RawCounts folder

    df = pd.read_csv(data_folder / 'fit_results.csv')
    fit_amp = df['Fitted Amplitude'].fillna(0)  # Fill NaN values with 0 for plotting
    in_to_um = 2.54 * 1e4 # convert inches to micrometers
    step_from_file = data_folder.parent.stem.split('_')[-1]  # get the characters after the last underscore in the folder name, excluding file extension
    try: 
        step_from_file = float(data_folder.parent.stem.rsplit('_', 1)[-1]) # get the characters after the last underscore in the folder name, excluding file extension
    except ValueError:
        step_from_file = 1e-4 # if it fails, set to 1e-4 as default
    step_to_um = in_to_um * step_from_file
    #print("Step size in micrometers: ", step_to_um)
    # one step is 1e-4 of an inch, one inch is 2.54cm and then multiply by 1e-4 to get to um 
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
    ax.plot(x_fit, erf_func(x_fit, *popt), color='red', label=f'Error Function Fit \nWidth: {round(popt[1], 2)}$\\pm$ {round(perr[1], 2)} μm') #, \nAmplitude: {round(popt[0], 2)}, \nCentre: {round(step_to_um * popt[2], 2)} μm, \nOffset: {round(popt[3], 2)}')
    #ax.set_xlim(50, 500)
    #ax.set_ylim(0, 3000)
    ax.set_xlabel('Knife Edge Position (μm)')
    ax.set_ylabel('Fitted Amplitude (Hz)')
    ax.set_title(f'Knife Edge Analysis: {data_folder.parent.name}/{data_folder.name}')
    ax.legend()
    print("ERF width for ", data_folder.parent.name,"/", data_folder.name, ": ", round(popt[1], 2), " ± ", round(perr[1], 2), " μm")
    #print("data root: ", data_folder.name)
    save_figure = "y" # input("y or n to save figure? ")
    if save_figure == "y":
        savefig = folder_dir / f'{data_folder.parent.name}_{data_folder.name}_erf_fit.png'
        fig.savefig(savefig, dpi=300)
    print("""\n ========================= \n """)