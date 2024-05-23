import numpy as np
import matplotlib.pyplot as plt
from map2curve import map2curve
from scipy.optimize import curve_fit

pi = np.pi

def function(ksi, *args):
    """
    Model function to fit the phase curves.

    Parameters:
    - ksi: array-like, the phase values.
    - args: tuple, the slice parameters.

    Returns:
    - curve: array-like, the fitted curve.
    """
    J = list(args)
    slice_num = len(J)
    pps = len(ksi) / slice_num
    curve = map2curve(slice_num, J, points_per_slice=pps, true_len=len(ksi), plot=False)
    return curve

def curve2map(phase_curves, full_phase, slice_num, plot=False, save=False):
    """
    Convert phase curves to N-slice maps.

    Parameters:
    - phase_curves: array-like, the phase curves (1D or 2D array).
    - full_phase: array-like, the full phase values.
    - slice_num: int, the number of slices.
    - plot: bool, whether to plot the fit results.
    - save: bool, whether to save the results to text files.

    Returns:
    - maps: 2D array, the N-slice maps.
    """
    # Generate longitude and phase arrays
    phi = np.linspace(-pi, pi, slice_num)  # longitude
    ksi = np.linspace(0, 2 * pi, len(full_phase))  # phase

    # Initialize arrays to store the maps and their errors
    maps = np.empty((0, slice_num))
    maps_err = np.empty((0, slice_num))

    # Initial parameters for the curve fitting
    init = [0.1] * slice_num
    bounds = ([0] * slice_num, [np.inf] * slice_num)

    # Ensure phase_curves is 2D for consistent processing
    if phase_curves.ndim == 1:
        phase_curves = phase_curves[:, np.newaxis]

    # Loop through each column (phase curve) in the input data
    for col_idx in range(phase_curves.shape[1]):
        phase_curve = phase_curves[:, col_idx]

        # Fit the phase curve
        popt, pcov = curve_fit(function, ksi, phase_curve, p0=init, bounds = bounds)

        # Extract the optimal parameters and their errors
        Js = popt
        Jerr = np.sqrt(np.diag(pcov))

        # Plot the fit results if requested
        if plot:
            fit = function(ksi, *Js)
            plt.plot(ksi, phase_curve, label='Original')
            plt.plot(ksi, fit, label='Best Fit')
            plt.legend()
            plt.show()

        # Append the results to the maps and maps_err arrays
        maps = np.vstack((maps, np.reshape(Js, (1, len(Js)))))
        maps_err = np.vstack((maps_err, np.reshape(Jerr, (1, len(Jerr)))))

    # Save the results to text files if requested
    if save:
        np.savetxt('Nslice_maps.txt', maps, fmt='%f', delimiter='\t')
        np.savetxt('Nslice_maps_err.txt', maps_err, fmt='%f', delimiter='\t')

    return maps, maps_err

# Example usage
if __name__ == "__main__":
    # Example data
    single_light_curve = np.random.rand(74)  # Example single light curve
    multiple_light_curves = np.random.rand(74, 10)  # Example multiple light curves
    full_phase = np.linspace(0, 2 * pi, 74)  # Full phase
    slice_num = 5  # Number of slices

    # Process single light curve
    maps_single, error = curve2map(single_light_curve, full_phase, slice_num, plot=True, save=False)


    # Process multiple light curves
    maps_multiple, errors = curve2map(multiple_light_curves, full_phase, slice_num, plot=True, save=False)

