import numpy as np
import matplotlib.pyplot as plt
from map2curve import map2curve
from astropy.table import Table
from scipy.optimize import curve_fit
pi = np.pi

def function(ksi, *args):
    J = list(args)
    slice_num = len(J)
    pps = len(ksi) / slice_num
    curve = map2curve(slice_num, J, points_per_slice=pps, true_len=len(ksi), plot=False)
    return curve

def curve2map(phase_curves, full_phase, slice_num, plot=False, save=False):
    phi = np.linspace(-pi, pi, slice_num)  # longitude
    ksi = np.linspace(0, 2 * pi, len(full_phase))  # phase
    maps = np.empty((0, slice_num))
    maps_err = np.empty((0, slice_num))

    init = [0] * slice_num  # Initialize with zeros

    # Check if phase_curves is 1D or 2D, and handle accordingly
    if phase_curves.ndim == 1:
        phase_curves = phase_curves[:, np.newaxis]

    for col_idx in range(phase_curves.shape[1]):
        phase_curve = phase_curves[:74, col_idx]

        popt, pcov = curve_fit(function, ksi, phase_curve, p0=init)

        Js = popt
        Jerr = np.sqrt(np.diag(pcov))

        if plot:
            fit = function(ksi, *Js)
            plt.plot(ksi, phase_curve, label='original')
            plt.plot(ksi, fit, label='best fit')
            plt.legend()
            plt.show()

        maps = np.vstack((maps, np.reshape(Js, (1, len(Js)))))
        maps_err = np.vstack((maps_err, np.reshape(Jerr, (1, len(Jerr)))))

    if save:
        np.savetxt('Nslice_maps.txt', maps, fmt='%f', delimiter='\t')
        np.savetxt('Nslice_maps_err.txt', maps_err, fmt='%f', delimiter='\t')

    return maps
