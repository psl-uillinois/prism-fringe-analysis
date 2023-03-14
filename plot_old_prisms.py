import numpy as np
import matplotlib.pyplot as plt

lps = [20, 22.5, 25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50]
index_avg = [1.36, 1.43747, 1.46189, 1.47789, 1.48968, 1.504, 1.51495, 1.52421, 1.53684, 1.54526, 1.55537, 1.56379, 1.57642]
index_stderr = [0.04463, 0.02358, 0.01684, 0.01347, 0.01347, 0.01347, 0.01516, 0.016, 0.01684, 0.01852, 0.01684, 0.01516, 0.01516]
index_stdev = [x*(10**(1/2)) for x in index_stderr]
print(index_stdev)

# Selectable by user
data_file = "output_index.csv"
x_axis = "Laser power (mW)"
y_axis = "Refractive index"

plt.figure(1, figsize=(4, 2))
font = {'size'   : 8}
plt.rc('font', **font)
plt.subplot(1, 2, 1)
#plt.plot(lps, index_avg)
plt.ylim([1.15, 1.65])
plt.gca().tick_params(which='both', direction="in")
#plt.scatter([lp/2 for lp in lps], np.array(index_stdev)/(np.array(index_avg) - 1.12))
plot1 = plt.errorbar([lp/2 for lp in lps], index_avg, index_stdev, linestyle='None', marker='o', capsize=3, color='k')
#plt.gca().yaxis.set_visible(False)
#plt.gca().set_yticklabels([])
#plt.title("Data from literature")
plt.xlabel(x_axis)
plt.ylabel(y_axis)

# Plot new prisms for comparison

prism_data = np.transpose(np.genfromtxt(data_file, delimiter=' '))

plt.subplot(1, 2, 2)
plt.ylim([1.15, 1.65])
plt.gca().tick_params(which='both', direction="in")
#plt.scatter(prism_data[0]/2, prism_data[2]/(prism_data[1] - 1.15))
if np.shape(prism_data)[0] == 3:
    plt.errorbar(prism_data[0]/2, prism_data[1], prism_data[2], linestyle='None', marker='o', capsize=3, ms=1.5, color='k')
elif np.shape(prism_data)[0] == 2:
    plt.scatter(prism_data[0], prism_data[1], color='k')
else:
    print("Error: Prism data malformed")
    exit(1)


plt.xlabel(x_axis)
plt.ylabel(y_axis)
#plt.title("Calibrated prisms")
#plt.legend(["Old data (LSA paper)", "New data (Calibrated prisms)"])

#plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()

plt.savefig('prism_data.png', dpi=1200)

plt.show()
