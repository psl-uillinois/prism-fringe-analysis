import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

fringes_1 = np.transpose(np.genfromtxt("output_fringes_custom.csv", delimiter=' '))
fringes_2 = np.transpose(np.genfromtxt("output_fringes_custom_.csv", delimiter=' '))
index_1 = np.transpose(np.genfromtxt("output_index_custom.csv", delimiter=' '))
index_2 = np.transpose(np.genfromtxt("output_index_custom_.csv", delimiter=' '))

plt.figure(1, figsize=(4, 5))
font = {'size'   : 15}
plt.rc('font', **font)
#plt.scatter(index_1[0], index_1[1], s=2)
#plt.scatter(index_2[0], index_2[1], s=2, marker='^')
#plt.legend(["Control", "Constant time"])
#plt.xlabel("Z (mm)")
#plt.ylabel("Index")
#plt.xlim([0, 2.4])
#plt.ylim([1, 1.5])
#plt.savefig('z_scan_idx.png', dpi=1200)
#plt.figure(2, figsize=(8, 6))
plt.scatter(fringes_1[0], fringes_1[1], s=5)
plt.scatter(fringes_2[0], fringes_2[1], s=5, marker='s')
lgd = plt.legend(["Control", "Constant time"], markerscale=4, loc=2, prop={'size': 12})
plt.xlabel("Z (mm)")
plt.ylabel("Fringe spacing ($\mu$m)")
plt.xlim([0.4, 1.0])
plt.ylim([5, 15])
plt.savefig('z_scan.png', dpi=1200)

# Show to user
plt.show()
