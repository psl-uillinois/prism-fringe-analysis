import sys

import numpy as np
import scipy
import scipy.signal
import scipy.interpolate
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import median_filter

# === HELPER FUNCTIONS ===

# https://doi.org/10.1021/ac051370e
# Implemented from:
# "Baseline Correction with Asymmetric Least Squares Smoothing"
# by Paul H. C. Eilers Hans F.M. Boelens
# October 21, 2005
def als_baseline(data, lamb, p, order=2, eps=1e-12, max_iter=1000):
    size = np.max(np.shape(data))
    diff_matrix = np.diff(np.eye(size), order)
    unsmooth_penalty = scipy.sparse.csc_matrix(lamb * (diff_matrix @ diff_matrix.T))
    baseline = 0
    last_baseline = None
    for i in range(0, max_iter):
        weights = np.where(data > baseline, p, 1 - p)
        fit_penalty = scipy.sparse.diags(weights, format="csc")
        baseline = scipy.sparse.linalg.spsolve(fit_penalty + unsmooth_penalty,
                                               fit_penalty * data)
        if last_baseline is not None:
            if np.max(np.abs(baseline - last_baseline)) < eps:
                break
        last_baseline = baseline
    return baseline


def get_index_fringe_spacing(ynew, prom, max_peaks=None):
    peaks, _ = scipy.signal.find_peaks(ynew, prominence=prom)  # Use for custom script

    if max_peaks is not None:
        while len(peaks) > max_peaks and len(peaks) > 1:
            if ynew[peaks[-1]] > ynew[peaks[0]]:
                peaks = peaks[1:]
            else:
                peaks = peaks[:-1]

    if len(peaks) <= 1:
        return [0, []]

    peaks = np.array(peaks)

    peak_width = np.median(np.diff(peaks))
    return [peak_width, peaks]

def single_get_fringe_spacing(x_data, y_data, prom, max_peaks):
    y_filtered = median_filter(y_data, size=5)
    [initial_peak_width, peaks] = get_index_fringe_spacing(y_filtered, prom)
    if initial_peak_width != initial_peak_width:
        initial_peak_width = 11 / 1.5
    window = int(initial_peak_width * 1.5)

    if window % 2 == 0:
        window += 1

    if window <= 10:
        window = 11

    polyorder = min(5, window - 1)
    savgol = scipy.signal.savgol_filter(y_data, window, polyorder)
    savgol = savgol - als_baseline(savgol, lamb=10000, p=0.0001) # Alt: 1000, 0.0001

    savgol_interp = scipy.interpolate.interp1d(x_data, savgol, kind='cubic')
    xnew = np.linspace(x_data[0], x_data[-1], num=100000, endpoint=True)
    ynew = savgol_interp(xnew)

    [peak_width, peaks] = get_index_fringe_spacing(ynew, prom, max_peaks=max_peaks)

    fringe_spacing = peak_width * (xnew[1] - xnew[0])
    return fringe_spacing


# Retrieves fringe spacing in um given raw data and scaling factors
# prism_data: 2D array [y, x] of collected light intensities at points in space
# x_scale: Distance in um between samples in x (e.g., between prism_data[0, 0] and prism_data[0, 1])
# y_scale: Distance in um between samples in y (e.g., between prism_data[0, 0] and prism_data[1, 0])
def get_fringe_spacing(prism_data, prom=50, max_peaks=None):
    x_data = np.arange(0, np.shape(prism_data)[1])
    averaged_along_y = np.mean(prism_data, axis=0)

    return single_get_fringe_spacing(x_data, averaged_along_y, prom, max_peaks=max_peaks)

# === MAIN FUNCTION ===

if __name__ == "__main__":
    full_metadata = sys.stdin.read().strip().replace('\r', '\n').split('\n')
    absolute_filename = full_metadata[0]
    min_idx = None
    max_idx = None
    max_peaks = None
    for i in range(1, len(full_metadata)):
        single_metadata = full_metadata[i]
        if single_metadata.isnumeric():
            single_metadata = int(single_metadata)
            if min_idx is None:
                min_idx = single_metadata
            elif max_idx is None:
                max_idx = single_metadata
            else:
                max_peaks = single_metadata
                break

    imdata = np.loadtxt(absolute_filename)[:, 1:2]
    prism_data = np.transpose(np.array(imdata))[:, min_idx:max_idx]
    fringe_spacing = get_fringe_spacing(prism_data, max_peaks=max_peaks)
    sys.stdout.write(f"{fringe_spacing}\n")
    sys.stdout.flush()
