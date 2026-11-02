import numpy as np
import pandas as pd
from scipy import special
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys
from pathlib import Path

from scripts.lorentzian_fit import load_csv, lorentzian, fit_lorentzian # analyse spectra of pillars
from scripts.erf_fit import erf_func, fit_erf # analyse knife edge amplitude points

import sys
from pathlib import Path
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent  # ...\LG05-ExperimentalData\KnifeEdge\Code
    
except NameError:
    PROJECT_ROOT = Path.cwd().parent # jup
sys.path.append(str(PROJECT_ROOT))

#data_dir = PROJECT_ROOT.parent / "2608" / "260812" / "C6C-200-4A_FourierEdge_1e-4"
#folder_root = PROJECT_ROOT.parent / "2608" / "260818" # input("Enter the path to the data directory: ")
folder_dir = PROJECT_ROOT.parent / "2610" / "261027"   # input("Enter the path to the data directory: ")

print(f"Searching for data folders in ", str(folder_dir))
for data_folder in folder_dir.iterdir(): # search for * subfolders in folder_dir (daily folder) which have a PowerNorm subfolder
    if data_folder.is_dir(): # check its a folder (directory)
        files = [f for f in data_folder.glob("*.csv") if 'results' not in f.name.lower()] 
        # search for all .csv files in data_folder (pillar measurement) which don't contain 'results' in their filename
        if files:
            earliest_file = min(files, key=lambda f: f.stat().st_mtime)
            #print("Earliest File: ", earliest_file.name)
            print(len(files), "matching CSV files found in ", data_folder.parent.name, data_folder.name)
        else:
            print("No matching CSV files found in ", data_folder.parent.name, data_folder.name)
#print(str(folder_dir.name))
#print("data folder parent: ", data_folder.parent)
print("Found", len(list(folder_dir.iterdir())), "data folders in ", str(folder_dir))

#folder_dir = PROJECT_ROOT.parent / "2608" / "test_folder_iter" # global folder which has subfolders. each global is one per day
#print("Searching for folders in ", str(folder_dir))

"""

folder_dir = PROJECT_ROOT.parent / "2608" / "260818"
data_folder.parent.name = C6C-200-4A_FourierEdge_1e-4
data_folder.name = PowerNorm or RawCounts


folder_dir / data_folder.parent / data_folder / .csv files with counts data

"""

print("""\n ========================= \n """)
# for data_folder in folder_dir.glob("*/" + data_norm_yn): # search for subfolders (i.e. pillar measurement folders) in folder_dir (daily folder) which have a PowerNorm subfolder
#     if data_folder.is_dir(): # check its a folder (directory)
#         print(f"Found {data_folder.name} folder in: ", data_folder.parent.name)
#         files = [f for f in data_folder.glob("*.csv") if 'results' not in f.name.lower() ]
#         if files:
#             earliest_file = min(files, key=lambda f: f.stat().st_mtime)
#             print("Earliest File: ", earliest_file.name)
#             print(len(files), "matching CSV files found in ", data_folder.name)
#         else:
#             print("No matching CSV files found in ", data_folder.name)
master_csv = [] # master csv file to hold all pillar results from all days. Contains erf width and error, NA and error
#master_row = [] # store results that will be appended to master csv per folder (pillar)
for data_folder in folder_dir.iterdir(): # find all subfolders in the daily global folder 
    print(data_folder.name)
    if not data_folder.is_dir(): # skip anything that isn't a folder
        print("Skipping non-directory: ", data_folder.name)
        continue
    # try:
    #     pillar_id = str(data_folder.name.split('_')[1]) # get the pillar ID from the folder name. take the [0] index of the parent folder name when splitting by underscore
    #     pillar_diam = float(pillar_id.split('-')[1][:3])/100 # get the pillar size from the folder name. take the [1] index of the parent folder name and then the first 3 characters
    #     if pillar_id == '':
    #         pillar_id = "Unknown"
    #     if pillar_diam == '':
    #         pillar_diam = np.nan
    # except ValueError:
    #     print("Error: Could not extract pillar ID from folder name.")
    #     pillar_id = "Unknown"
    #     pillar_diam = np.nan
    pillar_id = "Unknown"
    pillar_diam = np.nan
    laser_measurement = False
    bigsquare_measurement = False
    if 'BIGSQUARE' in data_folder.parent.name.upper():
        bigsquare_measurement = True
        continue
    if 'laser' in data_folder.parent.name.lower():
        laser_measurement = True
        print("Laser measurement detected.")
        #print("Using peak value minus mean outside the window instead of fitted amplitude for laser measurements.")

        
    #print("pillar ID: ", pillar_id)
    #print(f"Found {data_folder.name} folder in: ", data_folder.parent.name)
    
    files = [f for f in data_folder.glob("*.csv") if 'results' not in f.name.lower()] 
    # 
    #data_dir = folder
    #print("data dir: ", data_dir)
    if files:
        earliest_file = min(files, key=lambda f: f.stat().st_mtime)
        #print(len(files), "matching CSV files found")
    else:
        print("No matching CSV files found in ")
        continue

    rows = [] # to hold results to put in csv file for each data folder
    master_csv_row = {
        'Folder': data_folder.parent.name,
        'Pillar ID': pillar_id,
        'Pillar diameter': pillar_diam,
        } # this variable will contain results for each column of master csv file

    lam, counts = load_csv(earliest_file) # load data from .csv
    fig0, ax= plt.subplots()
    ax.plot(lam, counts, label='Raw Data')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Counts')
    ax.set_title(f'Raw Data for {data_folder.name}')
    ax.legend()
    fig0.savefig(data_folder / f'{data_folder.name}_raw_data.png', dpi=300)
    peaks, props = find_peaks(counts, 
                              prominence=100, 
                              distance=8, 
                              height=2000 - np.min(counts)) # find peaks
    #print(f"File: {earliest_file.name}, {len(peaks)} peaks found at wavelengths: {lam[peaks]}")
    lam_fundamental = lam[peaks][-1] # get the wavelength of the fundamental peak
    counts_fundamental = counts[peaks][-1] # get the counts of the fundamental peak
    print("Fundamental wavelength: ", lam_fundamental)
    #print("Fundamental counts for data folder ", data_folder.parent.name,"/", data_folder.name, ": ", counts_fundamental)
    


    files_peakless = []
    filenum = 0
    for f in files:
        row = {'File': f.name}
        filenum += 1
        try:
            lam, counts = load_csv(f)
            window = 25
            lam_window = (lam > lam_fundamental - window/2) & (lam < lam_fundamental + window/2)
            lam_windowed = lam[lam_window]
            counts_windowed = counts[lam_window]
            counts_outside_window = counts[~lam_window]
            #print(f"File: {f.name}, \nFitting window: {lam_windowed.min()} - {lam_windowed.max()} nm")
            peaks, props = find_peaks(
                counts_windowed, 
                prominence=100, 
                distance=60, 
                height=50,
                )
            #print(f"Filenum: {filenum-1}, File: {f.name}, {len(peaks)} peaks found at wavelengths: {lam_windowed[peaks]}")

            if len(peaks) == 0:
                #print("No peaks found in file: ", f.name, "in data folder: ", data_folder.name, " - skipping onto next file")
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

            if laser_measurement == True:
                row.update({
                'Fitted Amplitude': counts_windowed[peaks][0] - np.min(counts_outside_window) if len(peaks) > 0 else np.nan,
                'Amplitude Error': perr[0],
                'Fitted Wavelength': popt[2],
                'Wavelength Error': perr[2],
                'Fitted Width': popt[1],
                'Width Error': perr[1],
                'Fitted Offset': popt[3],
                'Offset Error': perr[3],
                'Fit Status': 'Peak value used'
            })
                #print(f"Using peak value of {counts_windowed[peaks][0] - np.min(counts_outside_window)} for amplitude instead of fitted amplitude for this measurement.")
                laser_plot='n'
                if laser_plot=='y':
                #if 200 <= filenum < 205:
                    #print("trying to plot for file with ", len(peaks), " peaks found at wavelengths: ", lam_windowed[peaks])
                    fig, ax = plt.subplots()
                    ax.plot(lam, counts, label='Raw Data')
                    ax.plot(lam_windowed[peaks][0], counts_windowed[peaks][0], 'x', label='Peak for ERF Plot')

                    #ax.plot(lam_windowed, y_fit, label='Lorentzian Fit', color='red')
                    ax.axvspan(lam_windowed.min(), lam_windowed.max(), color='yellow', alpha=0.3, label='Fitting Window')
                    ax.set_xlabel('Wavelength (nm)')
                    ax.set_ylabel('Counts')
                    ax.set_title(f'Data with peak for {f.name}')
                    ax.legend()
                    plt.show()
            else:

                lower_bounds = [0.3 * (counts_windowed[peaks][0]-np.min(counts_windowed)), 2, lam_fundamental - 5, 0.1*np.min(counts_windowed)] # amp, width, wavelength, offset
                upper_bounds = [1.5 * (counts_windowed[peaks][0]-np.min(counts_windowed)), 10, lam_fundamental + 5, 5 * np.min(counts_windowed)] # amp, width, wavelength, offset
                #print("Lower bound amplitude: ", lower_bounds[0])
                #print("Upper bound amplitude: ", upper_bounds[0])
                #print(len(files_peakless), "files with no peaks found in ", data_folder.name)
                guesses = [{
                    "amplitude": counts_windowed[peaks][0] - np.min(counts_windowed),
                    "width": 10,
                    "wavelength": lam_fundamental,
                    "offset": np.min(counts_windowed)
                }]
                #print(f"Filenum: {filenum-1}, Guessed amplitude: ", guesses[0]["amplitude"])
                try:
                    popt, perr, y_fit = fit_lorentzian(
                        lam_windowed,
                        counts_windowed, 
                        guesses,
                        bounds=(lower_bounds, upper_bounds)
                        )
                except Exception as exc:
                    print(f"Fitting failed for {f.name}")
                    print(repr(exc))
                    popt, perr, y_fit = None, None, None
                    row.update({
                        'Fitted Amplitude': np.nan,
                        'Amplitude Error': np.nan,
                        'Fitted Wavelength': np.nan,
                        'Wavelength Error': np.nan,
                        'Fitted Width': np.nan,
                        'Width Error': np.nan,
                        'Fitted Offset': np.nan,
                        'Offset Error': np.nan,
                        'Fit Status': 'Fit failed'
                    })

                    rows.append(row)
                    continue
                p='n'
                if p=='y':
                #if 0 <= filenum < 5:
                    #print("trying to plot for file with ", len(peaks), " peaks found at wavelengths: ", lam_windowed[peaks])
                    fig, ax = plt.subplots()
                    ax.plot(lam, counts, label='Raw Data')
                    ax.plot(lam_windowed[peaks][0], counts_windowed[peaks][0], 'x', label='Peak for Fitting')

                    ax.plot(lam_windowed, y_fit, label='Lorentzian Fit', color='red')
                    ax.axvspan(lam_windowed.min(), lam_windowed.max(), color='yellow', alpha=0.3, label='Fitting Window')
                    ax.set_xlabel('Wavelength (nm)')
                    ax.set_ylabel('Counts')
                    ax.set_title(f'Lorentzian Fit for {f.name}')
                    ax.legend()
                    plt.show()
                
                #print(f"File: {f.name}, Fitted parameters: amplitude, width, wavelength, offset: ")
                #print(popt)


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

            



        except Exception as exc: # if something that isn't the fitting raises an exception, do this
            print(f"Unexpected error for {f.name}")
            print(repr(exc))
            row.update({
                'Fitted Amplitude': counts_windowed[peaks][0] - np.mean(counts_outside_window) if len(peaks) > 0 else np.nan,
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
    filenum = 0 # reinitialise for next data folder
    print(len(files_peakless), "files with no peaks found out of", len(files), "total files.") # how many files had no peaks in *Pillar measurement* / PowerNorm or RawCounts folder

    results = pd.DataFrame(rows)
    
    results.to_csv(data_folder / f'fit_results_{data_folder.name}.csv', index=False) # save results to PowerNorm or RawCounts folder

    ### Analyse amplitudes

    ke_image_size = 2800 * 100 / 500 #fourier image diameter at knife edge in um

    df = pd.read_csv(data_folder / f'fit_results_{data_folder.name}.csv')
    fit_amp = df['Fitted Amplitude'].fillna(0)  # Fill NaN values with 0 for plotting
    final_cavity_width = df['Fitted Width'].iloc[-1]  
    #in_to_um =  2.54 * 2.54 * 1e4 # convert inches to micrometers
    try: 
        stepfile = data_folder.name.split("_")[2] # get
        print("stepfile:", stepfile)
        step_from_file = float(stepfile.removesuffix("um"))   # 50.0  (Python 3.9+)
        print("Step size from folder name: ", step_from_file)
    except ValueError:
        step_from_file = 2e-3 # if it fails, set to 5e-4 as default
        print("Could not parse step size from folder name, using default: ", step_from_file)
    step_to_um = 1 * step_from_file
    #print("Step size in micrometers: ", step_to_um)
    # one step is 1e-4 of an inch, one inch is 2.54cm and then multiply by 1e-4 to get to um 
    ke_pos = step_to_um * np.arange(len(df))

    erf_guess = [{
        "amplitude": np.max(fit_amp),
        "width": -50,
        "centre": np.argmin(np.abs(fit_amp - np.max(fit_amp) / 2)),  # Use the index of where the amp is closest to half the max amp as the centre guess
        "offset": fit_amp.min()
    }]
    try:
        popt, perr, erf_fit = fit_erf(ke_pos, fit_amp, erf_guess)
        #print("guessing centre position as:", ke_pos[erf_guess[0]["centre"]])

    except Exception as exc:
        print(f"Error occurred while fitting ERF to file {data_folder.name}")
        print(repr(exc))
        popt, perr, erf_fit = None, None, None





    x_fit = np.linspace(np.min(ke_pos), np.max(ke_pos), 1000)

    fig, ax = plt.subplots()
    ax.plot(ke_pos, fit_amp, marker='o', linestyle='-', color='blue')
    if popt is not None:
        erf_width = abs(popt[1])
        na = 0.7 * (2 * erf_width / (0.5 * ke_image_size)) # 0.7 to correct for objective max NA=0.7
        ax.plot(x_fit, erf_func(x_fit, *popt), color='red', label=f'ERF Fit \nWidth: {round(erf_width, 2)}$\\pm$ {round(perr[1], 2)} μm \nNA: {round(na, 3)}$\\pm$ {round(2 * perr[1] / (0.5 * ke_image_size), 3)}') #, \nAmplitude: {round(popt[0], 2)}, \nCentre: {round(step_to_um * popt[2], 2)} μm, \nOffset: {round(popt[3], 2)}')
        print("ERF width for", pillar_id, ": ", round(erf_width, 2), " ± ", round(perr[1], 2), " μm")

        #print("2 * width / Fourier image size: ",round(2*erf_fit / (0.5 * ke_image_size), 2))
    #ax.set_xlim(50, 500)
    #ax.set_ylim(0, 3000)
    ax.set_xlabel('Knife Edge Position (μm)')
    ax.set_ylabel('Fitted Amplitude (Hz)')
    ax.set_title(f'Knife Edge Analysis: {data_folder.name}')
    ax.legend()
    #print("data root: ", data_folder.name)
    save_figure = "y" # input("y or n to save figure? ")
    if save_figure == "y":
        #savefig = folder_dir / f'{data_folder.name}_erf_fit.png'
        savefig = Path(r"C:\Users\et20091\temp") / f'{data_folder.name}_erf_fit.png'
        fig.savefig(savefig, dpi=300)
        
    master_csv_row.update({ # add these columns to each row of the master csv
        'Fitted ERF Width (um)': erf_width if popt is not None else np.nan,
        'ERF Width Error (um)': perr[1] if perr is not None else np.nan,
        'Fitted NA': 2 * erf_width / (0.5 * ke_image_size) if popt is not None else np.nan,
        'NA Error': 2 * perr[1] / (0.5 * ke_image_size) if perr is not None else np.nan,
        'Pillar Amplitude': counts_fundamental,
        'Pillar Wavelength (nm)': lam_fundamental,
        'Pillar Cavity Width (nm)': final_cavity_width,
    })

    # Save results for each folder (pillar) to master csv file and plot distribution of fits vs pillar size

    master_csv.append(master_csv_row) # append row with results to master csv file
    master_results = pd.DataFrame(master_csv)
    #master_csv_location = PROJECT_ROOT.parent / '2609' / '260929' 
    master_csv_name = f'master_pillar_results_{folder_dir.name}.csv'
    master_results.to_csv(folder_dir / master_csv_name, index=False) # save results to PowerNorm or RawCounts folder
    print("""\n ========================= \n """)

print("Saved erf fits and NAs to", master_csv_name, "in", folder_dir)

### Plot NA vs pillar diameter from master csv file


