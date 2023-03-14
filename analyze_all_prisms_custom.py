from analyze_single_prism import get_fringe_spacing
import numpy as np
import os
import string
import matplotlib.pyplot as plt
import math
from scipy.interpolate import CubicSpline
from scipy.interpolate import pchip
from PIL import Image

# Selectable by user
input_data_dir = "20220531_INT20_CO"
prism_data_filenames = []  # Leave blank to be auto-filled by all files in directory
independent_variable = ""  # Determine type of independent variable used (laser power/lp or intensity/int)

# The following are only relevant when convert_to_index = True:
convert_to_index = True  # False outputs fringe spacing, true outputs index conversion
measured_wavelength = 0.642  # In microns, the laser source used for the measurement
background_index = 1.1517  # Measured separately by ellipsometry
z_top = 20.13
scale = 0.2743

if len(prism_data_filenames) == 0:
    # Retrieves all files in directory
    prism_data_filenames = [f for f in os.listdir(input_data_dir) if os.path.isfile(os.path.join(input_data_dir, f))]

output_data = []
output_index_data = []
# Iterate over all selected files
for filename in prism_data_filenames:
    if 'tiff' not in filename:
        continue
    # Retrieve metadata stored in filename
    # Note: Because periods cannot be easily used in a filename, "p" is used for decimal points in metadata.
    metadata = filename.split("_")

    # Laser power or intensity for a specific prism is stored as 00lp or 00int, respectively
    #lp_data = [x.lower() for x in metadata if independent_variable in x.lower()]
    #if len(lp_data) > 1:
    #    lp_data = [lp_data[0]]
    #if len(lp_data) != 1:
    #    print("Warning: " + filename + " has unreadable " + independent_variable + " metadata")
    #    continue
    #lp_data = float(lp_data[0].strip(string.ascii_lowercase).replace("p", "."))

    #print(metadata)
    z_data = float(metadata[6])
    z_loc = z_top - z_data
    #if z_data != 19.56:
    #    continue

    #if abs(z_loc - 1.52) > 0.005:
    #    continue

    # Degree metadata can be stored either directly (00deg) or as the thickness of the prism (always 100 um wide)
    deg_data = 15

    # Size is stored as 00x00um typically
    #physical_size = [x for x in metadata if ("x" in x and "um" in x)]
    #if len(physical_size) != 1:
    #    print("Warning: " + filename + " has unreadable physical size metadata")
    #    continue
    #physical_size = physical_size[0].split(".")[0].strip(string.ascii_lowercase).split("x")
    #physical_size = [float(x.replace("p", ".")) for x in physical_size]

    # Retrieve the raw data file
    #prism_data = np.genfromtxt(os.path.join(input_data_dir, filename), delimiter='\t')
    imdata = Image.open(os.path.join(input_data_dir, filename))
    prism_data = np.transpose(np.array(imdata))[160:370,:]
    #prism_data = np.array(imdata)

    # Pass to fringe analysis program
    fringe_spacing = get_fringe_spacing(prism_data, scale, scale, prom=50)

    if fringe_spacing == 0:
        continue

    # Apply index conversion formula if requested
    if convert_to_index:
        #calculated_index = measured_wavelength / (2 * math.radians(deg_data) * fringe_spacing) + background_index
        calculated_index = math.sqrt(background_index ** 2 * math.sin(math.radians(deg_data)) ** 2 +
                                     (measured_wavelength / 2 / fringe_spacing +
                                      background_index * math.sin(2*math.radians(deg_data))/2)**2 /
                                      math.sin(math.radians(deg_data)) ** 2)
        output_index_data.append((z_loc, calculated_index))

    # Save data entry
    output_data.append((z_loc, fringe_spacing))

# Separate data by independent variable (if multiple entries for a single design are present)
final_output_x = []
final_output_y = []

final_index_output_x = []
final_index_output_y = []

for data in output_data:
    if data[0] in final_output_x:
        idx = final_output_x.index(data[0])
        final_output_y[idx].append(data[1])
    else:
        final_output_x.append(data[0])
        final_output_y.append([data[1]])

if convert_to_index:
    for data in output_index_data:
        if data[0] in final_index_output_x:
            idx = final_index_output_x.index(data[0])
            final_index_output_y[idx].append(data[1])
        else:
            final_index_output_x.append(data[0])
            final_index_output_y.append([data[1]])

# Retrieve standard deviation and mean for each independent variable
# Mean is stored back into final_output_
final_output_y_stdev = []
final_index_output_y_stdev = []

fringe_points_x = []
fringe_points_y = []
for i in range(0, len(final_output_y)):
    data = final_output_y[i]
    for j in data:
        fringe_points_x.append(final_output_x[i])
        fringe_points_y.append(j)
    #final_output_y_stdev.append(np.std(data))
    final_output_y[i] = data[0]

index_points_x = []
index_points_y = []
# Separately, statistics will be computed for refractive index if requested
if convert_to_index:
    for i in range(0, len(final_index_output_y)):
        data = final_index_output_y[i]
        for j in data:
            index_points_x.append(final_index_output_x[i])
            index_points_y.append(j)
        #final_index_output_y_stdev.append(np.std(data))
        final_index_output_y[i] = data[0]

#print(final_index_output_y)
#print(final_index_output_y_stdev)

# Plot the final data -- chosen between four different possible datasets
#if not convert_to_index:
#    plt.scatter(final_output_x, final_output_y)
#else:
#    plt.scatter(final_index_output_x, final_index_output_y)

# Save data to CSV
# This data can be read later with plot_all_prisms.py
np.savetxt("output_fringes_custom.csv", np.transpose([fringe_points_x, fringe_points_y]), fmt="%.5f", delimiter=" ")

if convert_to_index:
    np.savetxt("output_index_custom.csv", np.transpose([index_points_x, index_points_y]), fmt="%.5f", delimiter=" ")

plt.figure(1, figsize=(8, 6))
font = {'size'   : 15}
plt.rc('font', **font)
plt.scatter(index_points_x, index_points_y, s=1)
plt.xlabel("Z ($\mu$m)")
plt.ylabel("Index")
plt.xlim([0, 2.4])
plt.ylim([1, 1.5])
plt.savefig('z_scan_idx.png', dpi=1200)
plt.figure(2, figsize=(8, 6))
plt.scatter(fringe_points_x, fringe_points_y, s=1)
plt.xlabel("Z ($\mu$m)")
plt.ylabel("Fringe spacing ($\mu$m)")
plt.xlim([0, 2.4])
plt.ylim([0, 100])
plt.savefig('z_scan.png', dpi=1200)

# Show to user
plt.show()
