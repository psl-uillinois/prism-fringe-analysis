import numpy as np
import matplotlib.pyplot as plt

lps = np.array([20, 22.5, 25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50])
index_avg = np.array([1.36, 1.43747, 1.46189, 1.47789, 1.48968, 1.504, 1.51495, 1.52421, 1.53684, 1.54526, 1.55537, 1.56379, 1.57642])
index_stderr = np.array([0.04463, 0.02358, 0.01684, 0.01347, 0.01347, 0.01347, 0.01516, 0.016, 0.01684, 0.01852, 0.01684, 0.01516, 0.01516])
index_stdev = index_stderr*(10**(1/2))
print(index_stdev)

# Selectable by user
data_file = "output_index.csv"
x_axis = "Laser power (mW)"
y_axis = "Refractive index"

plt.figure(1, figsize=(3, 1.375))
font = {'size'   : 8}
plt.rc('font', **font)
#plt.subplot(1, 2, 2)
#plt.plot(lps, index_avg)
#plt.ylim([1.2, 1.65])
#plt.scatter([lp/2 for lp in lps], np.array(index_stdev)/(np.array(index_avg) - 1.12))
#plot1 = plt.errorbar([lp/2 for lp in lps], index_avg, index_stdev, linestyle='None', marker='o', capsize=3, color='k')
#plt.gca().yaxis.set_visible(False)
#plt.gca().set_yticklabels([])
#plt.title("Data from literature")
#plt.xlabel(x_axis)
plt.scatter(index_avg, index_stdev/(index_avg)*100, color='royalblue', s=5)
#plt.scatter(index_avg[2:], index_stdev[2:]/(index_avg[2:])*100, color='royalblue')
#plt.errorbar(index_avg, index_stdev/(index_avg)*100, xerr=index_stdev, linestyle='None', marker='o', capsize=3, color='royalblue')
#plt.errorbar(index_avg[:2], index_stdev[:2]/(index_avg[:2])*100, xerr=index_stdev[:2], linestyle='None', marker='o', capsize=3, color='royalblue')
plt.ylabel(y_axis)

# Plot new prisms for comparison

prism_data = np.transpose(np.genfromtxt(data_file, delimiter=' '))

#plt.subplot(1, 2, 1)
#plt.ylim([1.2, 1.65])
#plt.scatter(prism_data[0]/2, prism_data[2]/(prism_data[1] - 1.15))
plt.scatter(prism_data[1], prism_data[2]/(prism_data[1])*100, color='r', marker='s', s=5)
#plt.errorbar(prism_data[1], prism_data[2]/(prism_data[1])*100, xerr=prism_data[2], linestyle='None', color='r', capsize=3, marker='s')

plt.yscale("log")
plt.xlabel("Refractive Index")
plt.ylabel("Coefficient of\nVariation (%)")
plt.ylim([0.04, 20])
#plt.ylim([-0.5, 12])
plt.xlim([1.15, 1.6])
ax = plt.gca()
ax.tick_params(which='both', direction="in")
#ax.axes.xaxis.set_ticklabels([])
#ax.axes.yaxis.set_ticklabels([])
#plt.tick_params(
#    axis='x',          # changes apply to the x-axis
#    which='both',      # both major and minor ticks are affected
#    bottom=False,      # ticks along the bottom edge are off
#    top=False,         # ticks along the top edge are off
#    labelbottom=False) # labels along the bottom edge are off
#plt.legend(["Old data (LSA paper)", "New data (Calibrated prisms)"])

plt.subplots_adjust(left=0.25, right=0.95, top=0.99, bottom=0.25)
#plt.tight_layout()

plt.savefig('toc_figure.png', dpi=2400, bbox_inches='tight')

plt.show()
