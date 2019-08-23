import numpy as np
def binspec(wvl, flux, uncer, bin_factor):
    
    """
    Parameters
    ----------
    wvl : np.ndarray
        Original logspaced bin edges.
    flux : np.ndarray
        Old fluxes.
    uncer : np.ndarray
        Old uncertainties.
    bin_factor : float
        Factor to reduce original resolution by.
        
    Returns
    -------
    new_wvl : np.ndarray
        New logspaced bin edges.
    new_flux : np.ndarray
        New fluxes.
    new_uncer : np.ndarray
        New uncertainties.
    """
    
    wvllog = np.log(wvl)
    number_of_newbins = float(len(wvllog)) / bin_factor
    if float(number_of_newbins).is_integer():
        N = bin_factor - 0.000000000000001
    else:
        N = bin_factor
    integer_bin_factor = int(N)    
    integer_bin_factor_2 = int(N)
    remaining_bin_fraction = N - integer_bin_factor
    remaining_bin_fraction_2 = N - integer_bin_factor
    i = 0
    k = 0
    old_bin_length = wvllog[i + 1] - wvllog[i]
    old_bin_radius = float(old_bin_length) / 2.0
    new_bin_length = old_bin_length * N
    new_bin_radius = float(new_bin_length) / 2.0
    bin_radius_diff = new_bin_radius - old_bin_radius
    new_wvl = []
    new_flux = []
    new_uncer = []
    new_bin_center = wvllog[i] + bin_radius_diff
    new_wvl.append(np.exp(new_bin_center))
    new_flux.append(flux[i:i + integer_bin_factor].sum() + remaining_bin_fraction * flux[i + integer_bin_factor])
    new_uncer.append(np.sqrt(np.sum(np.square(uncer[i:i + integer_bin_factor])) + np.square(remaining_bin_fraction * uncer[i + integer_bin_factor])))
    while i + integer_bin_factor + (int(N - 1 + remaining_bin_fraction) + 1) <= len(wvllog) - 1:
        i = i + integer_bin_factor
        k = k + integer_bin_factor_2
        integer_bin_factor = int(N - (1 - remaining_bin_fraction)) + 1
        integer_bin_factor_2 = int(N - (1 - remaining_bin_fraction_2)) + 1
        remaining_bin_fraction_2 = N - (integer_bin_factor - remaining_bin_fraction_2)
        new_bin_center +=  (1 - remaining_bin_fraction) * (wvllog[i+1] - wvllog[i])  + wvllog[i + integer_bin_factor] - wvllog[i + 1] + remaining_bin_fraction_2 * (wvllog[i + integer_bin_factor] - wvllog[i + integer_bin_factor - 1])
        new_wvl.append(np.exp(new_bin_center))
        new_flux.append((1 - remaining_bin_fraction) * flux[i] + flux[i + 1:i + integer_bin_factor].sum() + remaining_bin_fraction_2 * flux[i + integer_bin_factor])
        new_uncer.append(np.sqrt(np.square((1 - remaining_bin_fraction) * uncer[i]) + np.sum(np.square(uncer[i + 1:i + integer_bin_factor])) + np.square(remaining_bin_fraction_2 * uncer[i + integer_bin_factor])))
        remaining_bin_fraction = N - (integer_bin_factor - remaining_bin_fraction)
    return np.array(new_wvl), np.array(new_flux), np.array(new_uncer)