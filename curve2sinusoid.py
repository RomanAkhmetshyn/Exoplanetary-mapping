import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from scipy.optimize import curve_fit
from math import ceil

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100


def map_coefficient(j, mode_type):
    if j == 1:
        return 4 / np.pi if mode_type == 'C' else -4 / np.pi
    else:
        if mode_type == 'C':
            return 1 / (-2 / (j**2 - 1) * (-1)**(j // 2)) * 2
        elif mode_type == 'D':
            return 1 / (2 / (j**2 - 1) * (-1)**(j // 2)) * 2

def sinusoid_map(phi, modes, include_odd_modes=False):
    """
    Generate a special sequence based on the number of modes specified.

    Parameters:
    - phi: array-like, the phase values.
    - modes: int, the number of modes.
    - include_odd_modes: bool, whether to include odd modes (except mode 1).

    Returns:
    - sequence: a function representing the generated sequence.
    - sequence_str: the generated sequence as a string.
    """
    terms = ["F0"]
    for mode in range(1, modes + 1):
        if include_odd_modes or mode == 1 or mode % 2 == 0:

            coef_C = map_coefficient(mode, 'C')
            coef_D = map_coefficient(mode, 'D')
            terms.append(f"{coef_C} * C{mode} * np.cos({mode} * phi)")
            terms.append(f"{coef_D} * D{mode} * np.sin({mode} * phi)")

    sequence_str = " + ".join(terms)
    sequence_code = f"lambda phi, F0, {', '.join([f'C{i}, D{i}' for i in range(1, modes + 1) if include_odd_modes or i == 1 or i % 2 == 0])}: {sequence_str}"
    
    sequence = eval(sequence_code)
    
    return sequence

def Fourier(ksi, modes, include_odd_modes=False):
    """
    Generate a sequence based on the number of modes specified.

    Parameters:
    - ksi: array-like, the phase values.
    - modes: int, the number of modes.
    - include_odd_modes: bool, whether to include odd modes (except mode 1).

    Returns:
    - sequence: a function representing the generated sequence.
    - sequence_str: the generated sequence as a string.
    """
    terms = ["F0"]
    for mode in range(1, modes + 1):
        if include_odd_modes or mode == 1 or mode % 2 == 0:
            # print(mode)
            terms.append(f"C{mode} * np.cos({mode} * ksi)")
            terms.append(f"D{mode} * np.sin({mode} * ksi)")

    sequence_str = " + ".join(terms)
    sequence_code = f"lambda ksi, F0, {', '.join([f'C{i}, D{i}' for i in range(1, modes + 1) if include_odd_modes or i == 1 or i % 2 == 0])}: {sequence_str}"
    

    
    sequence = eval(sequence_code)
    
    return sequence

def curve2sinusoid(phase_curves, full_phase, longitude_slices, max_mode, 
                   include_odd_modes=False, 
                   plot=False, 
                   best_fit=False,
                   curve_err=None):
    phi = np.linspace(-np.pi, np.pi, longitude_slices) #longitude
    ksi = np.linspace(0, 2*np.pi, len(full_phase)) #phase
    maps = np.empty((0, len(phi)))
    
    fits = []
    coef_table = []
    coef_errors = []
    
    Fur_function = Fourier(ksi, max_mode, include_odd_modes)
    Map_function = sinusoid_map(phi, max_mode, include_odd_modes)

    init = np.zeros(1+max_mode*2)
    if not include_odd_modes:
        init = np.zeros(1+max_mode*2-(ceil(max_mode/2)-1)*2)
        
    if phase_curves.ndim == 1:
        phase_curves = phase_curves[:, np.newaxis]

    # Loop through each column (phase curve) in the input data
    for col_idx in range(phase_curves.shape[1]):
        phase_curve = phase_curves[:, col_idx]
        
        
        if curve_err is not None:
            popt, pcov = curve_fit(Fur_function, ksi, phase_curve, p0=init,
                                   sigma=curve_err, absolute_sigma = True)
        else:
            popt, pcov = curve_fit(Fur_function, ksi, phase_curve, p0=init)
        
        coefs = popt
        
        coef_table.append(coefs)
        
        # coefs = np.vstack((coefs, np.reshape(popt, (1, len(popt)))))
        
        coef_err = np.sqrt(np.diag(pcov))
        coef_errors.append(coef_err)

        if best_fit:
            fit = Fur_function(ksi, *coefs)
            fits.append(fit)
            
        if plot:
            fit = Fur_function(ksi, *coefs)
            plt.plot(ksi, phase_curve, label='Original')
            plt.plot(ksi, fit, label='Best Fit')
            plt.legend()
            plt.show()
            

            
            
        J = Map_function(phi, *coefs)
        maps = np.vstack((maps, np.reshape(J, (1, len(J)))))
        
    if best_fit:
        return maps, fits, coef_table, coef_errors
    else:
        return maps
            
if __name__ == "__main__":
    # Example data
    single_light_curve = np.random.rand(100)  # Example single light curve
    multiple_light_curves = np.random.uniform(0.4, 0.9, (100, 10))
    full_phase = np.linspace(0, 2 * np.pi, 100)  # Full phase
    slice_num = 5  # Number of slices

    # Process single light curve
    # curve2sinusoid(single_light_curve, full_phase, 36, 6, False, plot=True)


    # Process multiple light curves
    maps = curve2sinusoid(multiple_light_curves, full_phase, 36, 6, False, plot=False)
    
    plt.plot(np.linspace(-np.pi, np.pi, 36),maps.T)
    plt.xlabel('longitude [rad]')
    plt.ylabel(' flux')
    plt.legend()
    plt.show()

