# Analyse master pillar results for one diameter


import sys
import numpy as np
import pandas as pd
from scipy import special
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent  # ...\LG05-ExperimentalData\KnifeEdge\Code
    
except NameError:
    PROJECT_ROOT = Path.cwd().parent # jup
sys.path.append(str(PROJECT_ROOT))

# Define directory to analyse

folder_dir = PROJECT_ROOT.parent / "2611" / "261112" / "225"   

master_csv_name = f'{folder_dir.name}_master_pillar_results.csv'
df3 = pd.read_csv(folder_dir / master_csv_name)
print("Master CSV file read successfully from", master_csv_name, "in", folder_dir)
pillar_id = df3.iloc[:, 1].to_numpy()
pillar_diams = df3.iloc[:, 2].to_numpy()
pillar_erf_widths = df3.iloc[:, 3].to_numpy()
pillar_erf_width_error = df3.iloc[:, 4].to_numpy()
pillar_nas = df3.iloc[:, 5].to_numpy()
pillar_na_error = df3.iloc[:, 6].to_numpy()
pillar_amplitudes = df3.iloc[:, 7].to_numpy()
pillar_wavelengths = df3.iloc[:, 8].to_numpy()
pillar_cavity_widths = df3.iloc[:, 9].to_numpy()

pillar_block = pillar_id[0].split('-')[0]  # get the block name from the pillar id


distinct_pillar_id = list(set(pillar_id)) # find distinct pillar IDs
amplitude_avg_list, wavelength_avg_list, cavity_width_avg_list, diams_list, width_avg_list, width_error_avg_list, na_avg_list, na_error_avg_list = [], [], [], [], [], [], [], []
print("Distinct pillar IDs found: ", distinct_pillar_id)
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
    'Pillar ID': distinct_pillar_id,
    'Diameter': diams_list,
    'Average ERF Width (um)': width_avg_list,
    'Average ERF Width Error (um)': width_error_avg_list,
    'Average NA': na_avg_list,
    'Average NA Error': na_error_avg_list,
    'Average Amplitude': amplitude_avg_list,
    'Average Wavelength (nm)': wavelength_avg_list,
    'Average Cavity Width (nm)': cavity_width_avg_list,
})

avg_master_csv_name = folder_dir.parent.name + '_avg_' + master_csv_name
df_avg.to_csv(folder_dir / avg_master_csv_name, index=False)
print("Saved averaged multimode results to", avg_master_csv_name, "in", folder_dir)


fig, ax3 = plt.subplots()
ax3.plot(diams_list, na_avg_list, 'o', label='Avg NA', markersize=10)
ax3.errorbar(diams_list, na_avg_list, yerr=na_error_avg_list,
             fmt='none', ecolor='gray', alpha=0.5,
             capsize=3, capthick=1, elinewidth=1,
             label='NA Error')
ax3.set_xlabel('Pillar Diameter (um)')
ax3.set_ylabel('NA')
ax3.set_title(folder_dir.parent.name + ': ' + pillar_block + '-' + str(int(100*diams_list[0]))) # split the master csv name to remove the .csv extension and use the rest as the title
ax3.legend()
#savefig = (folder_dir / avg_master_csv_name).with_suffix('.png')
savefig = folder_dir / f'{folder_dir.parent.name}_{pillar_block}-{str(int(100*diams_list[0]))}_NA.png'
fig.savefig(savefig, dpi=300)
plt.show()


na_arr = np.array(na_avg_list)  # one averaged NA per distinct pillar ID

bin_width = 0.02
lo = np.floor(na_arr.min() / bin_width) * bin_width
hi = np.ceil(na_arr.max() / bin_width) * bin_width
bins = np.round(np.arange(lo, hi + bin_width / 2, bin_width), 10)  # round to avoid float drift in edges
if len(bins) < 2:  # all values fall exactly on one edge
    bins = np.array([lo, lo + bin_width])

fig, ax4 = plt.subplots()
ax4.hist(na_arr, bins=bins, edgecolor='black')
ax4.set_xlabel('Averaged NA')
ax4.set_ylabel('Number of pillars')
ax4.set_title(folder_dir.parent.name + ': ' + pillar_block + '-' + str(int(100*diams_list[0]))) # split the master csv name to remove the .csv extension and use the rest as the title
ax4.set_xticks(bins)
ax4.tick_params(axis='x', rotation=45)
fig.tight_layout()

savefig_hist = folder_dir / f'{folder_dir.parent.name}_{pillar_block}-{str(int(100*diams_list[0]))}_NA_hist.png'
fig.savefig(savefig_hist, dpi=300)
plt.show()

# Plot na / wavelength / amplitude / cavity width vs pillar ID for each distinct pillar ID

pillar_matrix = np.array([s.split('-')[-1] for s in distinct_pillar_id])
plot_type = 'NA' 
fig, ax5 = plt.subplots()
ax5.plot(pillar_matrix, na_avg_list, 'o', label='NA per pillar', markersize=10, color='blue')
#ax5.plot(distinct_pillar_id, wavelength_avg_list, 'o', label='Wavelength per pillar', markersize=10, color='red')
#ax5.plot(distinct_pillar_id, amplitude_avg_list, 'o', label='Amplitude per pillar', markersize=10, color='orange')
#ax5.plot(distinct_pillar_id, cavity_width_avg_list, 'o', label='Wavelength per pillar', markersize=10, color='green')
ax5.set_xlabel('Pillar ID')
ax5.set_ylabel(plot_type)
ax5.set_title(folder_dir.parent.name + ': ' + pillar_block + '-' + str(int(100*diams_list[0])) + ' ' + plot_type + ' individual' ) # split the master csv name to remove the .csv extension and use the rest as the title
ax5.tick_params(axis='x', rotation=45)
ax5.legend()
savefig = folder_dir / f'{folder_dir.parent.name}_{pillar_block}_{str(int(100*diams_list[0]))}um_{plot_type}_individual.png'
fig.savefig(savefig, dpi=300)
plt.show()
