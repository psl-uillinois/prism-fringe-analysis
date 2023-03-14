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
input_data_dir = "20220531_PrismsAll"
prism_data_filenames = []  # Leave blank to be auto-filled by all files in directory
independent_variable = "lp"  # Determine type of independent variable used (laser power/lp or intensity/int)
show_error_bars = True

# The following are only relevant when convert_to_index = True:
convert_to_index = True  # False outputs fringe spacing, true outputs index conversion
measured_wavelength = 0.643  # In microns, the laser source used for the measurement
background_index = 1.1517  # Measured separately by ellipsometry

scale = 0.2743

if len(prism_data_filenames) == 0:
    # Retrieves all files in directory
    prism_data_filenames = [f for f in os.listdir(input_data_dir) if os.path.isfile(os.path.join(input_data_dir, f))]

output_data = []
output_index_data = []
# Iterate over all selected files
for filename in prism_data_filenames:
    if "Cropped" not in filename:
        continue

    # Retrieve metadata stored in filename
    # Note: Because periods cannot be easily used in a filename, "p" is used for decimal points in metadata.
    metadata = filename.split("_")

    # Laser power or intensity for a specific prism is stored as 00lp or 00int, respectively
    lp_data = [x.lower() for x in metadata if independent_variable in x.lower()]
    if len(lp_data) > 1:
        lp_data = [lp_data[0]]
    if len(lp_data) != 1:
        print("Warning: " + filename + " has unreadable " + independent_variable + " metadata")
        continue
    lp_data = float(lp_data[0].strip(string.ascii_lowercase).replace("p", "."))

    # Laser power or intensity for a specific prism is stored as 00lp or 00int, respectively
    int_data = [x.lower() for x in metadata if 'int' in x.lower()]
    if len(int_data) > 1:
        int_data = [int_data[0]]
    if len(int_data) != 1:
        print("Warning: " + filename + " has unreadable " + independent_variable + " metadata")
        continue
    int_data = float(int_data[0].strip(string.ascii_lowercase).replace("p", "."))
    #if int_data != 20:
    #    continue

    # Degree metadata can be stored either directly (00deg) or as the thickness of the prism (always 100 um wide)
    deg_data = 15
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

    # Retrieve the raw data file
    #prism_data = np.genfromtxt(os.path.join(input_data_dir, filename), delimiter='\t')
    imdata = Image.open(os.path.join(input_data_dir, filename))
    #prism_data = np.array(imdata)
    prism_data = np.transpose(np.array(imdata))

    # Pass to fringe analysis program
    fringe_spacing = get_fringe_spacing(prism_data, scale, scale, prom=50, intensity=int(int_data))
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
        # if calculated_index > 1.315:
        #     print(metadata[2])
        #     continue
        output_index_data.append((lp_data, round(calculated_index, 5)))
        #print(f'({lp_data}, {round(calculated_index, 5)})', end=', ')

    # Save data entry
    output_data.append((lp_data, fringe_spacing))

#output_index_data = sorted(output_index_data, key=lambda x: x[0])
#output_index_data = list(zip(*output_index_data))
#print(output_index_data[0])
#print(output_index_data[1])
#exit(0)
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
    final_output_y_stdev.append(np.std(data))
    final_output_y[i] = np.mean(data)

index_points_x = []
index_points_y = []
# Separately, statistics will be computed for refractive index if requested
if convert_to_index:
    for i in range(0, len(final_index_output_y)):
        data = final_index_output_y[i]
        for j in data:
            index_points_x.append(final_index_output_x[i])
            index_points_y.append(j)
        final_index_output_y_stdev.append(np.std(data))
        final_index_output_y[i] = np.mean(data)

print(final_index_output_x)
print(final_index_output_y)
print(final_index_output_y_stdev)

# Plot the final data -- chosen between four different possible datasets
if not convert_to_index:
    if show_error_bars:
        plt.errorbar(final_output_x, final_output_y, final_output_y_stdev, linestyle='None', marker='o', capsize=3)
    else:
        plt.scatter(final_output_x, final_output_y)
else:
    if show_error_bars:
        plt.errorbar(final_index_output_x, final_index_output_y, final_index_output_y_stdev, linestyle='None', marker='o', capsize=3)
    else:
        plt.scatter(final_index_output_x, final_index_output_y)

# Save data to CSV
# This data can be read later with plot_all_prisms.py
if not show_error_bars:
    np.savetxt("output_fringes.csv", np.transpose([final_output_x, final_output_y]), fmt="%.5f", delimiter=" ")
else:
    np.savetxt("output_fringes.csv", np.transpose([final_output_x, final_output_y, final_output_y_stdev]), fmt="%.5f", delimiter=" ")

if convert_to_index:
    if not show_error_bars:
        np.savetxt("output_index.csv", np.transpose([final_index_output_x, final_index_output_y]), fmt="%.5f", delimiter=" ")
    else:
        np.savetxt("output_index.csv", np.transpose([final_index_output_x, final_index_output_y, final_index_output_y_stdev]), fmt="%.5f", delimiter=" ")

# Add labels to graph
if independent_variable == "lp":
    plt.xlabel("Laser power (%)")
elif independent_variable == "int":
    plt.xlabel("Intensity (arb.)")

if not convert_to_index:
    plt.ylabel("Fringe spacing (um)")
else:
    plt.ylabel("Refractive index")
plt.title("Prism data")

if convert_to_index:
    final_index_output_x, final_index_output_y = zip(*sorted(zip(final_index_output_x, final_index_output_y)))
    #fit = np.polyfit(final_index_output_x, final_index_output_y, 3)
    # interp = pchip(final_index_output_x, final_index_output_y)
    # print(interp)
    # #print(fit)
    # fit_x = np.linspace(final_index_output_x[0], final_index_output_x[-1], 101)
    # plt.plot(fit_x, interp(fit_x))

    plt.figure(2)
    plt.scatter(index_points_x, index_points_y, s=1)
else:
    plt.figure(2)
    plt.scatter(fringe_points_x, fringe_points_y, s=1)

# Show to user
plt.show()
