# Analyse averaged master csv for multiple diameters. point at daily folder, not size folder

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

folder_dir = PROJECT_ROOT.parent / "2611"    # input("Enter the path to the data directory: ")

avg_master_csv_name = f'{folder_dir.name}_avg_multi_master_pillar_results.csv'
df4 = pd.read_csv(folder_dir / avg_master_csv_name)
avg_pillar_id = df4.iloc[:, 0].to_numpy()
avg_pillar_diams = df4.iloc[:, 1].to_numpy()
avg_pillar_erf_widths = df4.iloc[:, 2].to_numpy()
avg_pillar_erf_width_error = df4.iloc[:, 3].to_numpy()
avg_pillar_nas = df4.iloc[:, 4].to_numpy()
avg_pillar_na_error = df4.iloc[:, 5].to_numpy()
avg_pillar_amplitudes = df4.iloc[:, 6].to_numpy()
avg_pillar_wavelengths = df4.iloc[:, 7].to_numpy()
avg_pillar_cavity_widths = df4.iloc[:, 8].to_numpy()

print("Master CSV file read successfully from", avg_master_csv_name, "in", folder_dir)

# ---- one colour per diameter, shared by both plots ----
# Colours are assigned in a fixed order (smallest diameter -> first colour), so a
# given diameter keeps the same colour in the scatter and in the histogram.
DIAM_COLOURS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                '#e87ba4', '#008300', '#6250d6', "#d3e348", "#e34948"]  # blue, orange, aqua, yellow, ...
unique_diams = np.unique(avg_pillar_diams)
if len(unique_diams) > len(DIAM_COLOURS):
    raise ValueError(f'{len(unique_diams)} distinct diameters but only {len(DIAM_COLOURS)} colours defined')
diam_colour = {d: DIAM_COLOURS[i] for i, d in enumerate(unique_diams)}
diam_label = {d: f'{d:g} µm' for d in unique_diams}


# ---- NA vs pillar diameter, coloured by diameter ----
fig, ax3 = plt.subplots()
for d in unique_diams:
    m = avg_pillar_diams == d
    
    ax3.errorbar(avg_pillar_diams[m], avg_pillar_nas[m], yerr=avg_pillar_na_error[m],
                 fmt='o', markersize=10, color=diam_colour[d],
                 markeredgecolor='white', markeredgewidth=1,   # white ring so overlapping points stay distinct
                 ecolor=diam_colour[d], elinewidth=1, capsize=3, capthick=1) 
                 #,label=diam_label[d])
    
# ---- mean ± spread of the pillars at each diameter ----
mean_na   = np.array([avg_pillar_nas[avg_pillar_diams == d].mean() for d in unique_diams])
n_pillars = np.array([(avg_pillar_diams == d).sum() for d in unique_diams])
sd_na     = np.array([avg_pillar_nas[avg_pillar_diams == d].std(ddof=1) if n > 1 else np.nan
                      for d, n in zip(unique_diams, n_pillars)])
sem_na    = sd_na / np.sqrt(n_pillars)

ax3.errorbar(unique_diams, mean_na, yerr=sd_na,      # swap sd_na for sem_na if you want the error on the mean
             fmt='_', markersize=22, markeredgewidth=2.5,
             color='black', ecolor='black', elinewidth=2, capsize=6, capthick=2,
             zorder=5, label='Mean ± SD')
ax3.set_xticks(unique_diams)       # tick only at the diameters that exist
ax3.margins(x=0.3)                 # keep the end points away from the axes edge
ax3.set_xlabel('Pillar Diameter (um)')
ax3.set_ylabel('NA')
ax3.set_ylim(0.28, 0.7)
ax3.set_xlim(1.8, 4.2)
ax3.set_title(avg_master_csv_name.split('.')[0] + ': C3')  # split the master csv name to remove the .csv extension and use the rest as the title
ax3.legend(title='Diameter')
#savefig = (folder_dir / avg_master_csv_name).with_suffix('.png')
savefig = folder_dir / f'{avg_master_csv_name.split(".")[0]}_C3.png'
#fig.savefig(savefig, dpi=300)



# ---- histogram of averaged NA, bars coloured by diameter ----
na_arr = np.array(avg_pillar_nas)  # one averaged NA per distinct pillar ID

bin_width = 0.05
lo = np.floor(na_arr.min() / bin_width) * bin_width
hi = np.ceil(na_arr.max() / bin_width) * bin_width
bins = np.round(np.arange(lo, hi + bin_width / 2, bin_width), 10)  # round to avoid float drift in edges
if len(bins) < 2:  # all values fall exactly on one edge
    bins = np.array([lo, lo + bin_width])

na_by_diam = [na_arr[avg_pillar_diams == d] for d in unique_diams]  # one NA array per diameter

fig, ax4 = plt.subplots()
ax4.hist(na_by_diam, bins=bins, stacked=False,   # stacked=False puts the diameters side by side within each bin instead
         color=[diam_colour[d] for d in unique_diams],
         label=[diam_label[d] for d in unique_diams],
         edgecolor='black',
         alpha=0.5) 
ax4.set_xlabel('Averaged NA')
ax4.set_ylabel('Number of pillars')
ax4.set_title(avg_master_csv_name.split('.')[0] + ': C3')
ax4.set_xticks(bins)
ax4.tick_params(axis='x', rotation=45)
ax4.legend(title='Diameter')
fig.tight_layout()

savefig_hist = folder_dir / f'{avg_master_csv_name.split(".")[0]}_C3_NA_hist.png'
#fig.savefig(savefig_hist, dpi=300)
