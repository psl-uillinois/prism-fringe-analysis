from analyze_single_prism import get_fringe_spacing
import numpy as np
import os
import string
import matplotlib.pyplot as plt
import math
from scipy.interpolate import CubicSpline
from scipy.interpolate import pchip

# Selectable by user
input_data_dir = "20221130_SinglePrisms"
prism_data_filenames = []  # Leave blank to be auto-filled by all files in directory
independent_variable = "lp"  # Determine type of independent variable used (laser power/lp or intensity/int)
show_error_bars = True

# The following are only relevant when convert_to_index = True:
convert_to_index = True  # False outputs fringe spacing, true outputs index conversion
measured_wavelength = 0.633  # In microns, the laser source used for the measurement
background_index = 1.1517  # Measured separately by ellipsometry


if len(prism_data_filenames) == 0:
    # Retrieves all files in directory
    prism_data_filenames = [f for f in os.listdir(input_data_dir) if os.path.isfile(os.path.join(input_data_dir, f))]

data_2 = []
pos_2 = []

output_data = []
output_index_data = []

output_data_ct = []
output_index_data_ct = []
# Iterate over all selected files
for filename in prism_data_filenames:
    # Retrieve metadata stored in filename
    # Note: Because periods cannot be easily used in a filename, "p" is used for decimal points in metadata.
    metadata = filename.split("_")
    # if int(metadata[-1].split('.')[0]) != 5:
    #     continue
    # Laser power or intensity for a specific prism is stored as 00lp or 00int, respectively
    lp_data = [x.lower() for x in metadata if independent_variable in x.lower()]
    if len(lp_data) > 1:
        lp_data = [lp_data[0]]
    if len(lp_data) != 1:
        print("Warning: " + filename + " has unreadable " + independent_variable + " metadata")
        continue
    lp_data = float(lp_data[0].strip(string.ascii_lowercase).replace("p", "."))

    # Degree metadata can be stored either directly (00deg) or as the thickness of the prism (always 100 um wide)
    deg_data = 0
    if convert_to_index:
        deg_data = [x.lower() for x in metadata if "deg" in x.lower()]
        if len(deg_data) != 1:
            deg_data = [x.lower() for x in metadata if "thick" in x.lower()]
            if len(deg_data) != 1:
                print("Warning: " + filename + " has unreadable angle metadata")
                continue
            else:
                deg_data = math.degrees(math.atan(float(deg_data[0].strip(string.ascii_lowercase).replace("p", ".")) / 50))
        else:
            deg_data = float(deg_data[0].strip(string.ascii_lowercase).replace("p", "."))

    # Size is stored as 00x00um typically
    physical_size = [x for x in metadata if ("x" in x and "um" in x)]
    if len(physical_size) != 1:
        print("Warning: " + filename + " has unreadable physical size metadata")
        continue
    physical_size = physical_size[0].split(".")[0].strip(string.ascii_lowercase).split("x")
    physical_size = [float(x.replace("p", ".")) for x in physical_size]

    # Retrieve the raw data file
    prism_data = np.genfromtxt(os.path.join(input_data_dir, filename), delimiter='\t')

    size_y = int(np.shape(prism_data)[0]*1.0)
    min_y = 0
    step_y = 2

    sub_position = []
    sub_index = []
    for i in range(0, size_y//step_y):
        sub_prism_data = prism_data[i * step_y + min_y:(i + 1) * step_y + min_y,:]

        # Pass to fringe analysis program
        fringe_spacing = get_fringe_spacing(sub_prism_data, physical_size[0] / np.shape(prism_data)[1], physical_size[1] / np.shape(prism_data)[0])
        if fringe_spacing == 0:
            continue

        # Apply index conversion formula if requested
        if convert_to_index:
            #calculated_index = measured_wavelength / (2 * math.radians(deg_data) * fringe_spacing) + background_index
            calculated_index = math.sqrt(background_index ** 2 * math.sin(math.radians(deg_data)) ** 2 +
                                         (measured_wavelength / 2 / fringe_spacing +
                                          background_index * math.sin(2*math.radians(deg_data))/2)**2 /
                                          math.sin(math.radians(deg_data)) ** 2)
            # print(filename, calculated_index)

            position=i*0.5*step_y
            sub_position.append(position)
            sub_index.append(calculated_index)

    if metadata[2].split('.')[0] != "CT":
        output_index_data.append((lp_data, np.std(sub_index)))
    else:
        output_index_data_ct.append((lp_data, np.std(sub_index)))


lp_1, index_std_1 = zip(*output_index_data)
lp_2, index_std_2 = zip(*output_index_data_ct)

lp_1 = np.array(lp_1)/2
lp_2 = np.array(lp_2)/2
index_std_1 = np.array(index_std_1)
index_std_2 = np.array(index_std_2)

np.savetxt("output_index_vs_y.csv", [lp_1, index_std_1, lp_2, index_std_2], fmt="%.5f",
           delimiter=" ")
