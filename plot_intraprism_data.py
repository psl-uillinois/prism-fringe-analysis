import numpy as np
import matplotlib.pyplot as plt

# Selectable by user
data_file = "output_index_vs_y_good.csv"
data_file_2 = "output_index_vs_y_bad.csv"
x_axis = "Laser power (mW)"
#y_axis = "Refractive index"
y_axis = "Coefficient of Variation (%)"

prism_data = np.transpose(np.genfromtxt(data_file, delimiter=' '))
plt.figure(1, figsize=(6, 3))
font = {'size'   : 8}
plt.rc('font', **font)
plt.subplot(1, 2, 2)
#plt.plot(lps, index_avg)
#plt.ylim([1.15, 1.65])
plt.ylim([0, 2.9])
plt.gca().tick_params(which='both', direction="in")
#plt.scatter(prism_data[0]/2, prism_data[2]/(prism_data[1] - 1.15))
plt.scatter(prism_data[0]/2, prism_data[2]/prism_data[1]*100, color='k')
# if np.shape(prism_data)[0] == 3:
#     plt.errorbar(prism_data[0]/2, prism_data[1], prism_data[2], linestyle='None', marker='o', capsize=3, ms=1.5, color='k')
# elif np.shape(prism_data)[0] == 2:
#     plt.scatter(prism_data[0], prism_data[1], color='k')
# else:
#     print("Error: Prism data malformed")
#     exit(1)
plt.xlabel(x_axis)
plt.ylabel(y_axis)

# Plot new prisms for comparison

prism_data = np.transpose(np.genfromtxt(data_file_2, delimiter=' '))

plt.subplot(1, 2, 1)
#plt.ylim([1.15, 1.65])
plt.ylim([0, 2.9])
plt.gca().tick_params(which='both', direction="in")
#plt.scatter(prism_data[0]/2, prism_data[2]/(prism_data[1] - 1.15))

plt.scatter(prism_data[0]/2, prism_data[2]/prism_data[1]*100, color='k')
# if np.shape(prism_data)[0] == 3:
#     plt.errorbar(prism_data[0]/2, prism_data[1], prism_data[2], linestyle='None', marker='o', capsize=3, ms=1.5, color='k')
# elif np.shape(prism_data)[0] == 2:
#     plt.scatter(prism_data[0], prism_data[1], color='k')
# else:
#     print("Error: Prism data malformed")
#     exit(1)


plt.xlabel(x_axis)
plt.ylabel(y_axis)
#plt.title("Calibrated prisms")
#plt.legend(["Old data (LSA paper)", "New data (Calibrated prisms)"])

#plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()

plt.savefig('prism_data_vs_y.png', dpi=1200)

plt.show()
