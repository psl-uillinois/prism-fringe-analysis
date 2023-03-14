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
input_data_dir = "Y_Full"
prism_data_filenames = []  # Leave blank to be auto-filled by all files in directory
independent_variable = "int"  # Determine type of independent variable used (laser power/lp or intensity/int)
show_error_bars = True

# The following are only relevant when convert_to_index = True:
convert_to_index = True  # False outputs fringe spacing, true outputs index conversion
measured_wavelength = 0.642  # In microns, the laser source used for the measurement
background_index = 1.1517  # Measured separately by ellipsometry

scale = 0.2743

if len(prism_data_filenames) == 0:
    # Retrieves all files in directory
    prism_data_filenames = [f for f in os.listdir(input_data_dir) if os.path.isfile(os.path.join(input_data_dir, f))]

data_2 = []
pos_2 = []

output_data = []
output_index_data = []
# Iterate over all selected files
for filename in prism_data_filenames:
    if "Full" in filename:
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

    if lp_data != 80:
        continue

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

    max_x = np.argmax(np.mean(prism_data, axis=0))
    center_y = prism_data[:,max_x]
    y_min = np.min(center_y) * 0.5 + np.max(center_y) * 0.5
    first_y = np.argmax(center_y-y_min>0)
    last_y = len(center_y) - np.argmax(center_y[::-1]-y_min>0)

    #size_y = int(np.shape(prism_data)[0]*0.7)
    #min_y = int(np.shape(prism_data)[0]*0.15)
    #min_y = first_y
    #size_y = last_y - first_y
    #size_y = int(80/0.274)
    #min_y = int((first_y + last_y) / 2 - size_y / 2)
    size_y = int(np.shape(prism_data)[0]*1.0)
    min_y = 0
    step_y = 1

    sub_position = []
    sub_index = []
    for i in range(0, size_y//step_y):
        sub_prism_data = prism_data[i * step_y + min_y:(i + 1) * step_y + min_y,:]

        # Pass to fringe analysis program
        fringe_spacing = get_fringe_spacing(sub_prism_data, scale, scale, prom=300)
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

            position=i*0.274*step_y
            sub_position.append(position)
            sub_index.append(calculated_index)
            output_index_data.append((lp_data, calculated_index))

        # Save data entry
        output_data.append((lp_data, fringe_spacing))

    sub_position = np.array(sub_position)
    sub_index = np.array(sub_index)


    if metadata[2].split('.')[0] == "CT":
        data_2 = sub_index
        pos_2 = sub_position
        continue

    cc = np.corrcoef(sub_position, sub_index)
    print(cc)
    plt.scatter(sub_position, sub_index, s=3)
    plt.scatter(pos_2, data_2, s=3)
    plt.xlabel('Y Position ($\mu$m)')
    plt.ylabel('Refractive Index')
    plt.legend(['Calibrated prism', 'Control prism'])
    plt.show()
    exit(0)

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
plt.ylim((1.2, 1.32))
# Save data to CSV
# This data can be read later with plot_all_prisms.py
if not show_error_bars:
    np.savetxt("output_fringes_vs_y.csv", np.transpose([final_output_x, final_output_y]), fmt="%.5f", delimiter=" ")
else:
    np.savetxt("output_fringes_vs_y.csv", np.transpose([final_output_x, final_output_y, final_output_y_stdev]), fmt="%.5f", delimiter=" ")

if convert_to_index:
    if not show_error_bars:
        np.savetxt("output_index_vs_y.csv", np.transpose([final_index_output_x, final_index_output_y]), fmt="%.5f", delimiter=" ")
    else:
        np.savetxt("output_index_vs_y.csv", np.transpose([final_index_output_x, final_index_output_y, final_index_output_y_stdev]), fmt="%.5f", delimiter=" ")

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
