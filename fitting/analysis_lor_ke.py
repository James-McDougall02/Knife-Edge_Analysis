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
folder_dir = PROJECT_ROOT.parent / "2608" / "260827" / "done" # Folder for measurements (one per day)
normalised = 0 # 0 for RawCounts, 1 for PowerNorm data folders
if normalised == 0:
    data_norm_yn = "RawCounts"
elif normalised == 1:
    data_norm_yn = "PowerNorm" 

print(f"Searching for {data_norm_yn} folders in ", str(folder_dir))
for data_folder in folder_dir.glob("*/" + data_norm_yn): # search for subfolders in folder_dir (daily folder) which have a PowerNorm subfolder
    if data_folder.is_dir(): # check its a folder (directory)
        #print(f"Found {data_folder.name} folder in: ", data_folder.parent.name, data_folder.parent.name)
        files = [f for f in data_folder.glob("*.csv") if 'results' not in f.name.lower() ]
        if files:
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            #print("Latest File: ", latest_file.name)
            print(len(files), "matching CSV files found in ", data_folder.parent.name, data_folder.name)
        else:
            print("No matching CSV files found in ", data_folder.name)
print("Found", len(list(folder_dir.glob("*/" + data_norm_yn))), f"{data_norm_yn} folders in ", str(folder_dir))
master_csv = [] # master csv file to hold all pillar results from all days. Contains erf width and error, NA and error

for data_folder in folder_dir.glob("*/" + data_norm_yn): # find all subfolders in the daily global folder 
    if not data_folder.is_dir(): # skip anything that isn't a folder
        print("Skipping non-directory: ", data_folder.name)
        continue

    pillar_id = str(data_folder.parent.name.split('_')[0]) # get the pillar ID from the folder name. take the [0] index of the parent folder name when splitting by underscore
    print(f"Found {data_folder.name} folder in: ", data_folder.parent.name)
    pillar_diam = float(data_folder.parent.name.split('-')[1][:3])/100 # get the pillar size from the folder name. take the [1] index of the parent folder name and then the first 3 characters
    files = [f for f in data_folder.glob("*.csv") if 'results' not in f.name.lower() ]
    #data_dir = folder
    #print("data dir: ", data_dir)
    if files:
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        print(len(files), "matching CSV files found in ", data_folder.parent.name,"/", data_folder.name)
    else:
        print("No matching CSV files found in ", data_folder.parent.name,"/", data_folder.name)

    rows = [] # to hold results to put in csv file
    master_csv_row = { # this variable will contain results for each column of master csv file
            'Folder': data_folder.parent.name,
            'Pillar ID': pillar_id,
            'Pillar diameter': pillar_diam,
            } 

    lam, counts = load_csv(latest_file) # load data from .csv
    peaks, props = find_peaks(counts, prominence=100, distance=60, height=1) # find peaks
    long_lam_peak_idx = np.argmax(counts[peaks]) # find peak with largest index (longest wavelength)
    lam_fundamental = lam[peaks][long_lam_peak_idx] # get the wavelength of the fundamental peak
    counts_fundamental = counts[peaks][long_lam_peak_idx] # get the counts of the fundamental peak
    print("Fundamental wavelength for data folder ", data_folder.parent.name,"/", data_folder.name, ": ", lam_fundamental)


    #print("Lower bound amplitude: ", lower_bounds[0])
    #print("Upper bound amplitude: ", upper_bounds[0])

    files_peakless = []
    filenum = 0
    for f in files:
        row = {'File': f.name}
        filenum += 1

        try:
            lam, counts = load_csv(f)
            window = 15
            lam_window = (lam > lam_fundamental - window/2) & (lam < lam_fundamental + window/2)
            lam_windowed = lam[lam_window]
            counts_windowed = counts[lam_window]
            #print(f"File: {f.name}, \nFitting window: {lam_windowed.min()} - {lam_windowed.max()} nm")
            peaks, props = find_peaks(
                counts_windowed, 
                prominence=50, 
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


            lower_bounds = [0.3 * (counts_windowed[peaks][0]-np.min(counts_windowed)), 2, lam_fundamental - 5, 0.1*np.min(counts_windowed)] # amp, width, wavelength, offset
            upper_bounds = [1.5 * (counts_windowed[peaks][0]-np.min(counts_windowed)), 40, lam_fundamental + 5, 5 * np.min(counts_windowed)] # amp, width, wavelength, offset
            guesses = [{
                "amplitude": counts_windowed[peaks][0] - np.min(counts_windowed),
                "width": 10,
                "wavelength": lam_fundamental,
                "offset": np.min(counts_windowed)
            }]





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

            plot = 'n' # input("y or n to plot? ")
            if plot == "y":
            #if 70 <= filenum < 75:
                fig, ax = plt.subplots()
                ax.plot(lam, counts, label='Raw Data')
                ax.plot(lam_windowed[peaks][0], counts_windowed[peaks][0], 'x', label='Peak for Fitting')
                ax.plot(lam_windowed, y_fit, label='Lorentzian Fit', color='red')
                ax.axvspan(lam_windowed.min(), lam_windowed.max(), color='yellow', alpha=0.3, label='Fitting Window')
                ax.set_xlabel('Wavelength (nm)')
                ax.set_ylabel('Counts')
                ax.set_title(f'Lorentzian Fit for {f.name}')
                ax.legend()
            
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



        except Exception as exc:
            print(f"Unexpected error for file {f.name}")
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
                'Fit Status': 'Fit failed'
            })

        rows.append(row)
        
        #print(len(files_peakless), "files with no peaks found in ", data_folder.name)
    filenum = 0
    print(len(files_peakless), "files with no peaks found in ", data_folder.parent.name, data_folder.name) # how many files had no peaks in *Pillar measurement* / PowerNorm or RawCounts folder

    results = pd.DataFrame(rows)
    results.to_csv(data_folder / f'fit_results_{data_folder.parent.name}.csv', index=False) # save results to PowerNorm or RawCounts folder

    ### Analyse amplitudes ###

    ke_image_size = 1000 * 2.8 * 150 / 500 #fourier image diameter at knife edge in um

    df = pd.read_csv(data_folder / f'fit_results_{data_folder.parent.name}.csv')
    fit_amp = df['Fitted Amplitude'].fillna(0)  # Fill NaN values with 0 for plotting
    final_cavity_width = df['Fitted Width'].iloc[-1]  # Save cavity width (nm) from last file to save to master csv
    in_to_um = 2.54 * 1e4 # convert inches to micrometers
    step_from_file = data_folder.parent.stem.split('_')[-1]  # get the characters after the last underscore in the folder name, excluding file extension
    try: 
        step_from_file = float(data_folder.parent.stem.rsplit('_', 1)[-1]) # get the characters after the last underscore in the folder name, excluding file extension
        print("Step size from folder name: ", step_from_file)
    except ValueError:
        step_from_file = 5e-4 # if it fails, set to 5e-4 as default
        print("Could not parse step size from folder name, using default: ", step_from_file)
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
    try:
        popt, perr, erf_fit = fit_erf(ke_pos, fit_amp, erf_guess)
    except Exception as exc:
        print(f"Error occurred while fitting ERF to file {data_folder.name}")
        print(repr(exc))
        popt, perr, erf_fit = None, None, None

    x_fit = np.linspace(np.min(ke_pos), np.max(ke_pos), 1000)

    fig, ax = plt.subplots()
    ax.plot(ke_pos, fit_amp, marker='o', linestyle='-', color='blue')
    if popt is not None:
        ax.plot(x_fit, erf_func(x_fit, *popt), color='red', label=f'Error Function Fit \nWidth: {round(popt[1], 2)}$\\pm$ {round(perr[1], 2)} μm \nNA: {round(2*popt[1] / (0.5 * ke_image_size), 3)} $\\pm$ {round(2 * perr[1] / (0.5 * ke_image_size), 3)}') #, \nAmplitude: {round(popt[0], 2)}, \nCentre: {round(step_to_um * popt[2], 2)} μm, \nOffset: {round(popt[3], 2)}')
    #ax.set_xlim(50, 500)
    #ax.set_ylim(0, 3000)
    ax.set_xlabel('Knife Edge Position (μm)')
    ax.set_ylabel('Fitted Amplitude (Hz)')
    ax.set_title(f'Knife Edge Analysis: {data_folder.parent.name}/{data_folder.name}')
    ax.legend()
    #print("ERF width for ", data_folder.parent.name,"/", data_folder.name, ": ", round(popt[1], 2), " ± ", round(perr[1], 2), " μm")
    #print("data root: ", data_folder.name)
    save_figure = "y" # input("y or n to save figure? ")
    if save_figure == "y":
        savefig = folder_dir / f'{data_folder.parent.name}_{data_folder.name}_erf_fit.png'
        fig.savefig(savefig, dpi=300)

    master_csv_row.update({ # add these columns to each row of the master csv
        'Fitted Width (um)': popt[1] if popt is not None else np.nan,
        'Width Error (um)': perr[1] if perr is not None else np.nan,
        'Fitted NA': 2 * popt[1] / (0.5 * ke_image_size) if popt is not None else np.nan,
        'NA Error': 2 * perr[1] / (0.5 * ke_image_size) if perr is not None else np.nan,
        'Pillar Amplitude': counts_fundamental,
        'Pillar Wavelength (nm)': lam_fundamental,
        'Pillar Cavity Width (nm)': final_cavity_width,
    })

    # Save results for each folder (pillar) to master csv file and plot distribution of fits vs pillar size

    master_csv.append(master_csv_row) # append row with results to master csv file
    master_results = pd.DataFrame(master_csv)
    master_csv_location = PROJECT_ROOT.parent / '2609' / '260928' 
    master_csv_name = f'master_pillar_results_260928.csv'
    master_results.to_csv( master_csv_location / master_csv_name, index=False) # save results to PowerNorm or RawCounts folder



    print("Saved erf fits and NAs to ", master_csv_name, " in ", master_csv_location)
    print("""\n ========================= \n """)
    #plt.show()

### Plot NA vs pillar diameter from master csv file

df2 = pd.read_csv(PROJECT_ROOT.parent / 'master_pillar_results.csv')
pillar_diams = df2.iloc[:, 1].to_numpy()
pillar_nas = df2.iloc[:, 4].to_numpy()
pillar_na_error = df2.iloc[:, 5].to_numpy()

fig, ax2 = plt.subplots()
ax2.plot(pillar_diams, pillar_nas, 'o', label='Fitted NAs', markersize=10)
ax2.errorbar(pillar_diams, pillar_nas, yerr=pillar_na_error, fmt='none', ecolor='gray', alpha=0.5, label='NA Error')
ax2.set_xlabel('Pillar Diameter (μm)')
ax2.set_ylabel('NA')
ax2.set_title('NA vs Pillar Diameter')
ax2.legend()
savefig = PROJECT_ROOT.parent / 'master_pillar_na_results.png'
fig.savefig(savefig, dpi=300)

"""
Open master csv file and average results for each distinct pillar ID. Save to new csv file with averaged results.
Plot data and save as png
"""

df3 = pd.read_csv(master_csv_location / master_csv_name)
pillar_id = df3.iloc[:, 1].to_numpy()
pillar_diams = df3.iloc[:, 2].to_numpy()
pillar_erf_widths = df3.iloc[:, 3].to_numpy()
pillar_erf_width_error = df3.iloc[:, 4].to_numpy()
pillar_nas = df3.iloc[:, 5].to_numpy()
pillar_na_error = df3.iloc[:, 6].to_numpy()
pillar_amplitudes = df3.iloc[:, 7].to_numpy()
pillar_wavelengths = df3.iloc[:, 8].to_numpy()
pillar_cavity_widths = df3.iloc[:, 9].to_numpy()


"""
search thru data and find rows where pillar id is the same. average the nas and errors for that pillar id. 
"""


distinct_pillar_id = list(set(pillar_id))
amplitude_avg_list, wavelength_avg_list, cavity_width_avg_list, diams_list, width_avg_list, width_error_avg_list, na_avg_list, na_error_avg_list = [], [], [], [], [], [], [], []

for i in distinct_pillar_id: # go through each distinct pillar id
    mask = pillar_id == i # mask is where pillar id matches distinct pillar id
    diam = pillar_diams[mask][0] # get the diameter for that pillar id. [0] gets first instance of that pillar id, as all should have same diam
    pillar_nas_for_id = pillar_nas[mask] # get all the NAs
    pillar_erf_widths_for_id = pillar_erf_widths[mask] # get all the widths
    pillar_erf_width_error_for_id = pillar_erf_width_error[mask] # get all the width errors
    pillar_na_error_for_id = pillar_na_error[mask] # get all na errors
    width_avg_list.append(np.average(pillar_erf_widths_for_id, weights=1/pillar_erf_width_error_for_id**2)) # weighted average the widths for that pillar id
    width_error_avg_list.append(np.sqrt(1/(np.sum(1/pillar_erf_width_error_for_id**2)))) # weighted average the width errors
    amplitude_avg_list.append(np.average(pillar_amplitudes[mask], weights=1/pillar_na_error[mask]**2)) # weighted average the amplitudes for that pillar id
    wavelength_avg_list.append(np.average(pillar_wavelengths[mask], weights=1/pillar_na_error[mask]**2)) # weighted average the wavelengths for that pillar id
    cavity_width_avg_list.append(np.average(pillar_cavity_widths[mask], weights=1/pillar_na_error[mask]**2)) # weighted average the cavity widths for that pillar id
    na_avg_list.append(np.average(pillar_nas[mask], weights=1/pillar_na_error[mask]**2)) # weighted average the NAs for that pillar id
    na_error_avg_list.append(np.sqrt(1/(np.sum(1/pillar_na_error[mask]**2)))) # weighted average the na errors
    diams_list.append(diam) # append the diameter for that pillar id to the list



df_avg = pd.DataFrame({
    'Diameter': diams_list,
    'Pillar ID': distinct_pillar_id,
    'Average ERF Width (um)': width_avg_list,
    'Average ERF Width Error (um)': width_error_avg_list,
    'Average NA': na_avg_list,
    'Average NA Error': na_error_avg_list,
    'Average Amplitude': amplitude_avg_list,
    'Average Wavelength (nm)': wavelength_avg_list,
    'Average Cavity Width (nm)': cavity_width_avg_list,
})

avg_master_csv_name = 'avg_' + master_csv_name
df_avg.to_csv(master_csv_location / avg_master_csv_name, index=False)



fig, ax3 = plt.subplots()
ax3.plot(diams_list, na_avg_list, 'o', label='Avg NA', markersize=10)
ax3.errorbar(diams_list, na_avg_list, yerr=na_error_avg_list,
             fmt='none', ecolor='gray', alpha=0.5,
             capsize=3, capthick=1, elinewidth=1,
             label='NA Error')
ax3.set_xlabel('Pillar Diameter (um)')
ax3.set_ylabel('NA')
ax3.set_title(avg_master_csv_name.split('.')[0]) # split the master csv name to remove the .csv extension and use the rest as the title
ax3.legend()
savefig = (master_csv_location / avg_master_csv_name).with_suffix('.png')
fig.savefig(savefig, dpi=300)