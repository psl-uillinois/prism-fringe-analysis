import numpy as np
from scipy.interpolate import pchip
import matplotlib.pyplot as plt
import math
from numpy.polynomial import Polynomial
from functools import cache
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

intensities = [6.0, 110.0, 140.0, 170.0, 180.0, 190.0, 20.0, 210.0, 50.0, 80.0]
indices = [1.15, 1.2889975, 1.3441183333333333, 1.4401933333333334, 1.5082991666666665, 1.5534683333333337, 1.1970875, 1.577805, 1.2401255555555555, 1.2600069999999997]

fit = Polynomial.fit(indices, intensities, deg=4)
xrange = np.linspace(1.15, 1.65, 100)

font = {'size'   : 15}
plt.rc('font', **font)
plt.figure(figsize=(4, 5))
plt.plot(xrange, fit(xrange), color='k')
intensities = [110.0, 140.0, 170.0, 180.0, 190.0, 20.0, 210.0, 50.0, 80.0]
indices = [1.2889975, 1.3441183333333333, 1.4401933333333334, 1.5082991666666665, 1.5534683333333337, 1.1970875, 1.577805, 1.2401255555555555, 1.2600069999999997]

fit = Polynomial.fit(indices, intensities, deg=3)
plt.plot(xrange, fit(xrange), color='b')
plt.scatter([1.15], [6.0], color='k')
plt.scatter(indices, intensities, color='k')
plt.ylabel("Fluorescence Intensity (arb.)")
plt.xlabel("Refractive Index")
#plt.xlim([1.15, 1.65])
#plt.ylim([10, 230])
plt.xlim([1.10, 1.65])
plt.ylim([0, 230])
plt.savefig("index_vs_intensity.png", dpi=1200)
plt.show()