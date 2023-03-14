from analyze_single_prism import get_fringe_spacing
import numpy as np
import os
import string
import matplotlib.pyplot as plt
import math
from scipy.interpolate import CubicSpline
from scipy.interpolate import pchip

lps = [20, 22.5, 25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50]
index_avg = [1.36, 1.43747, 1.46189, 1.47789, 1.48968, 1.504, 1.51495, 1.52421, 1.53684, 1.54526, 1.55537, 1.56379, 1.57642]
index_stderr = [0.04463, 0.02358, 0.01684, 0.01347, 0.01347, 0.01347, 0.01516, 0.016, 0.01684, 0.01852, 0.01684, 0.01516, 0.01516]
index_stdev = [x*(10**(1/2)) for x in index_stderr]

peak_intensity_const = 0.1272

print(index_stdev)

# Selectable by user
data_file = "output_index.csv"
x_axis = "Laser power (mW)"
x2_axis = "Peak intensity (TW/cm$^2$)"
y_axis = "Refractive index"


plt.figure(1, figsize=(5, 5))
font = {'size'   : 8}
plt.rc('font', **font)
plt.subplot(2, 2, 1)
ax1 = plt.gca()
ax2 = ax1.twiny()
#plt.plot(lps, index_avg)
ax1.set_ylim([1.15, 1.65])
ax1.set_xlim([8, 26])
ax2.set_xlim([8*peak_intensity_const, 26*peak_intensity_const])
ax1.tick_params(which='both', direction="in")
ax2.tick_params(which='both', direction="in")
#plt.scatter([lp/2 for lp in lps], np.array(index_stdev)/(np.array(index_avg) - 1.12))
plot1 = ax1.errorbar([lp/2 for lp in lps], index_avg, index_stdev, linestyle='None', marker='o', capsize=3, elinewidth=0.5, capthick=0.5, ms=1.5, color='k')
plot2 = ax2.errorbar(np.array([lp/2 for lp in lps])*peak_intensity_const, index_avg, index_stdev, linestyle='None', marker='o', capsize=3, elinewidth=0.5, capthick=0.5, ms=1.5, color='k')
#plt.gca().yaxis.set_visible(False)
#plt.gca().set_yticklabels([])
#plt.title("Data from literature")
ax1.set_xlabel(x_axis)
ax2.set_xlabel(x2_axis)
ax1.set_ylabel(y_axis)

# Plot new prisms for comparison

prism_data = np.transpose(np.genfromtxt(data_file, delimiter=' '))

plt.subplot(2, 2, 2)
ax1 = plt.gca()
ax2 = ax1.twiny()
ax1.set_ylim([1.15, 1.65])
ax1.set_xlim([9, 18])
ax2.set_xlim([9*peak_intensity_const, 18*peak_intensity_const])
ax1.tick_params(which='both', direction="in")
ax2.tick_params(which='both', direction="in")
#plt.scatter(prism_data[0]/2, prism_data[2]/(prism_data[1] - 1.15))
if np.shape(prism_data)[0] == 3:
    ax1.errorbar(prism_data[0]/2, prism_data[1], prism_data[2], linestyle='None', marker='o', capsize=3, elinewidth=0.5, capthick=0.5, ms=1.5, color='xkcd:bright orange')
    ax2.errorbar(prism_data[0]/2*peak_intensity_const, prism_data[1], prism_data[2], linestyle='None', marker='o', capsize=3, elinewidth=0.5, capthick=0.5, ms=1.5, color='xkcd:bright orange')
elif np.shape(prism_data)[0] == 2:
    plt.scatter(prism_data[0], prism_data[1], color='k')
else:
    print("Error: Prism data malformed")
    exit(1)


ax1.set_xlabel(x_axis)
ax2.set_xlabel(x2_axis)
ax1.set_ylabel(y_axis)

font = {'size'   : 8}
plt.rc('font', **font)

plt.subplot(2, 2, 4)
ax1 = plt.gca()
ax2 = ax1.twiny()
ax1.tick_params(which='both', direction="in")
ax2.tick_params(which='both', direction="in")
[lp_1, index_std_1, lp_2, index_std_2] = np.genfromtxt("output_index_vs_y.csv", delimiter=' ')
ax1.scatter(lp_2, index_std_2, color='k', marker='o', s=8)
ax1.scatter(lp_1, index_std_1, color='xkcd:bright orange', marker='D', s=8)
ax2.scatter(lp_2*peak_intensity_const, index_std_2, color='k', marker='o', s=8)
ax2.scatter(lp_1*peak_intensity_const, index_std_1, color='xkcd:bright orange', marker='D', s=8)
ax1.set_xlabel('Laser power (mW)')
ax2.set_xlabel(x2_axis)
ax1.set_ylabel('Index SD')
ax1.set_ylim([0, 0.025])
ax1.set_xlim([9, 18])
ax2.set_xlim([9*peak_intensity_const, 18*peak_intensity_const])
#plt.xlim([])
plt.legend(['Control (with CT)', 'Fully calibrated'])


plt.subplot(2, 2, 3)
plt.gca().tick_params(which='both', direction="in")

# 50int fully corrected vs 20int control
fringes_1 = np.transpose(np.genfromtxt("output_fringes_custom.csv", delimiter=' '))
fringes_2 = np.transpose(np.genfromtxt("output_fringes_custom_.csv", delimiter=' '))
index_1 = np.transpose(np.genfromtxt("output_index_custom.csv", delimiter=' '))
index_2 = np.transpose(np.genfromtxt("output_index_custom_.csv", delimiter=' '))

plt.scatter(index_1[0][::5], index_1[1][::5], s=8, marker='o', color='k')
plt.scatter(index_2[0][::5], index_2[1][::5], s=8, marker='D', color='xkcd:bright orange')
lgd = plt.legend(["Control", "Fully calibrated"])
plt.xlabel("Z (mm)")
plt.ylabel("Refractive index")
plt.xlim([0.25, 1.25])
plt.ylim([1.205, 1.33])
#plt.ylim([5, 15])

plt.tight_layout()
plt.subplots_adjust(left=0.15, wspace=0.6, hspace=0.5)

plt.savefig('fig5.png', dpi=1200)

# Show to user
plt.show()