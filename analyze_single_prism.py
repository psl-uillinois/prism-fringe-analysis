import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import scipy.interpolate
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
import math
import statistics


def gaus(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def lorentzian(x, a, x0, gam):
    return a * gam**2 / ( gam**2 + ( x - x0 )**2)


def als_baseline(data, lamb, p, order=2, eps=1e-12, max_iter=1000):
    size = np.max(np.shape(data))
    diff_matrix = np.diff(np.eye(size), order)
    unsmooth_penalty = scipy.sparse.csc_matrix(lamb * (diff_matrix @ diff_matrix.T))
    print(unsmooth_penalty)
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


def arpls_baseline(data, lamb, order=2, eps=1e-7, max_iter=1000, initial_baseline=0):
    size = np.max(np.shape(data))
    diff_matrix = np.diff(np.eye(size), order)
    unsmooth_penalty = scipy.sparse.csc_matrix(lamb * (diff_matrix @ diff_matrix.T))
    baseline = initial_baseline
    last_baseline = None
    for i in range(0, max_iter):
        d_minus = data[data < baseline]
        md = 0
        sd = 1
        if np.max(np.shape(d_minus)) > 0:
            md = np.mean(d_minus)
            sd = np.std(d_minus)

        logistic_weights = scipy.special.expit(-2 * ((data - baseline) - (-md + 2 * sd)) / sd)
        weights = np.where(data > baseline, logistic_weights, 1)

        fit_penalty = scipy.sparse.diags(weights, format="csc")
        baseline = scipy.sparse.linalg.spsolve(fit_penalty + unsmooth_penalty,
                                               fit_penalty * data)
        if last_baseline is not None:
            if np.max(np.abs(baseline - last_baseline)) < eps:
                break
        last_baseline = baseline
    return baseline


def get_index_fringe_spacing(ynew, prom):
    peaks, _ = scipy.signal.find_peaks(ynew, prominence=prom)  # Use for custom script

    if len(peaks) > 7:
        peaks = peaks[1:-1]
    if len(peaks) > 5:
        peaks = peaks[1:-1]
    # if len(peaks) > 4:
    #     peaks = peaks[1:-1]

    if len(peaks) <= 1:
        return [0, []]

    peaks = peaks.tolist()
    for peak in peaks.copy():
        if peak < 0.1 * np.shape(ynew)[0] or peak > 0.9 * np.shape(ynew)[0]:
            peaks.remove(peak)
    peaks = np.array(peaks)

    #peak_width = (peaks[-1] - peaks[0]) / (len(peaks) - 1)
    peak_width = np.median(np.diff(peaks))
    return [peak_width, peaks]

def single_get_fringe_spacing(x_data, y_data, prom, intensity):
    avg_value = (min(y_data) + max(y_data)) / 3
    # window = 51
    # try:
    #     window = int(np.mean(np.diff(np.argwhere(np.diff(np.sign(y_data - avg_value))).flatten())))
    # except ValueError:
    #     pass

    #if window < 10:
    #    return 0

    # window *= 2

    y_filtered = median_filter(y_data, size=5)
    [initial_peak_width, peaks] = get_index_fringe_spacing(y_filtered, prom)
    if initial_peak_width != initial_peak_width:
        initial_peak_width = 11 / 1.5
    window = int(initial_peak_width * 1.5)
    #window_sizes = {20: 130, 50: 80, 80: 50, 110: 40, 140: 30, 170: 20, 180: 10, 190: 10, 210: 10}

    #window = window_sizes[intensity]
    if window % 2 == 0:
        window += 1

    if window <= 10:
        window = 11

    #print(window*0.27)

    polyorder = min(5, window - 1)
    savgol = scipy.signal.savgol_filter(y_data, window, polyorder)
    savgol1 = savgol
    #savgol = y_data
    savgol = savgol - als_baseline(savgol, lamb=10000, p=0.0001) # Alt: 1000, 0.0001

    savgol_interp = scipy.interpolate.interp1d(x_data, savgol, kind='cubic')
    xnew = np.linspace(x_data[0], x_data[-1], num=100000, endpoint=True)
    ynew = savgol_interp(xnew)

    # prom = np.max(ynew)/20
    if intensity >= 180:
        xnew = xnew[len(ynew)*1//3:len(ynew)*2//3]
        ynew = ynew[len(ynew)*1//3:len(ynew)*2//3]
    [peak_width, peaks] = get_index_fringe_spacing(ynew, prom)

    #print(peak_width)
    #print(peaks)

    fringe_spacing = peak_width * (xnew[1] - xnew[0])
    #print(fringe_spacing)

    # Placeholder to view data
    #peak_vals = [ynew[i] for i in peaks]

    #n = len(peaks)  # the number of data
    #if n > 7:
    #    mean = sum(peaks * peak_vals) / n  # note this correction
    #    sigma = sum(peak_vals * (peaks - mean) ** 2) / n  # note this correction
    #    print(peaks)
    #    print(peak_vals)
    #    popt, pcov = curve_fit(lorentzian, peaks * (xnew[1] - xnew[0]), peak_vals, p0=[1, 50, 10])
    #    gaussian_fit = lorentzian(xnew, *popt)
        #plt.figure(1)
        #plt.imshow(prism_data)

    # plt.figure(1)
    # plt.plot(x_data, y_data)
    # plt.plot(x_data, savgol1)
    # plt.plot(x_data, als_baseline(savgol1, lamb=1000, p=0.0001))
    # plt.plot(xnew, ynew)
    # highlighted_x_data = [xnew[i] for i in peaks]
    # highlighted_y_data = [ynew[i] for i in peaks]
    # plt.scatter(highlighted_x_data, highlighted_y_data)
    # plt.legend(('Raw data', 'Smoothed data with baseline subtracted', 'Peaks detected'))
    # plt.xlabel('X Position ($\mu$m)')
    # plt.ylabel('Intensity (arb.)')
    # plt.gca().tick_params(direction='in')
    # plt.ylim([-0.01, 0.38])
    # plt.show()
    #exit()
    return fringe_spacing


def process_edges(edges):
    locations = np.diff(edges)
    valid_locations = max([np.mean(locations) * 0.9 < x < np.mean(locations) * 1.1 for x in locations])
    if not valid_locations:
        return 0
    else:
        return np.mean(locations)


def single_get_fringe_spacing_LG(x_data, y_data):
    y_data = y_data - als_baseline(y_data, lamb=1000, p=0.0001)
    y_window = np.linspace(min(y_data), max(y_data), num=101, endpoint=True)
    max_length = 2
    max_length_fixed = False
    spacings = []
    while len(spacings) < 3:
        if max_length > 2:
            max_length -= 1
            max_length_fixed = True
            spacings = []
        for y_val in y_window:
            rising_edges = np.argwhere(np.diff(np.sign(y_data - y_val)) > 0).flatten()
            if len(rising_edges) == max_length or (not max_length_fixed and len(rising_edges) > max_length):
                spacing = process_edges(rising_edges)
                if spacing > 0:
                    if len(rising_edges) != max_length:
                        spacings = []
                    max_length = len(rising_edges)
                    spacings.append(spacing)
    #print(spacings)
    #print(np.median(spacings))
    return np.median(spacings) * (x_data[1] - x_data[0])


def get_angle_offset(prism_data, x_scale, y_scale):
    x_data = np.arange(0, np.shape(prism_data)[1]) * x_scale
    top_data = np.mean(prism_data[0:9], axis=0)
    bottom_data = np.mean(prism_data[-10:-1], axis=0)
    top_idx = np.argmax(top_data)
    bottom_idx = np.argmax(bottom_data)
    plt.figure(1)
    plt.plot(x_data, top_data)
    plt.scatter(x_data[top_idx], top_data[top_idx])
    plt.plot(x_data, bottom_data)
    plt.scatter(x_data[bottom_idx], bottom_data[bottom_idx])
    plt.show()
    exit(0)
    return 0


# Retrieves fringe spacing in um given raw data and scaling factors
# prism_data: 2D array [y, x] of collected light intensities at points in space
# x_scale: Distance in um between samples in x (e.g., between prism_data[0, 0] and prism_data[0, 1])
# y_scale: Distance in um between samples in y (e.g., between prism_data[0, 0] and prism_data[1, 0])
def get_fringe_spacing(prism_data, x_scale, y_scale, prom=0.01, intensity=20):
    #if lp == 131:
    #    plt.imshow(prism_data)
    #    plt.show()
    #exit(0)
    x_data = np.arange(0, np.shape(prism_data)[1]) * x_scale
    averaged_along_y = np.mean(prism_data, axis=0)

    return single_get_fringe_spacing(x_data, averaged_along_y, prom, intensity)
    #peak_widths = []
    #for i in range(0, np.shape(prism_data)[0]):
    #    single_peak_width = single_get_fringe_spacing(x_data, prism_data[i])
    #    if single_peak_width > 0:
    #        peak_widths.append(single_peak_width)

    #peak_width = statistics.median(peak_widths)
    #print(len(peak_widths), peak_width)

    #return peak_width
