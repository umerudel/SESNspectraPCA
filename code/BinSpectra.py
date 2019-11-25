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

# Drops unused bins from right

def binspec_left(wvl, flux, uncer, bin_factor):
    
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

# Drops unused bins from left

def binspec_right(wvl, flux, uncer, bin_factor):
    wvl_log = np.log(wvl)
    number_of_newbins = float(len(wvl_log)) / bin_factor
    if float(number_of_newbins).is_integer():
        N = bin_factor - 0.000000000000001
    else:
        N = bin_factor
    N1 = int(N)
    N3 = int(N)
    #n = N - N1
    #n0 = N - N1
    #######################################################################################################
    integer_number_of_new_bins = int(len(wvl_log) / N)                                                    #
    used_bins = integer_number_of_new_bins * N                                                            #
    unpaired_bins = len(wvl_log) - used_bins                                                              #
    N1_skip = int(unpaired_bins)                                                                          #
    remove_bin_fraction = unpaired_bins - int(unpaired_bins)                                              #
    n_new = 1 - remove_bin_fraction                                                                       #
    n_new_rem = N - n_new                                                                                 #
    n_new_2 = n_new_rem - int(n_new_rem - 0.00000000000001)                                               #
    n_new_0 = n_new_rem - int(n_new_rem - 0.00000000000001)                                               #
    N1_new = int(np.ceil(n_new_rem - 0.00000000000001))                                                   #
    #######################################################################################################
    i = N1_skip
    k = N1_skip
    old_bin_length = (wvl_log[i + 1] - wvl_log[i])              
    old_bin_radius = float(old_bin_length) / 2.0                 
    new_bin_length = old_bin_length * N                          
    new_bin_radius = float(new_bin_length) / 2.0                 
    bin_radius_diff = new_bin_radius - old_bin_radius
    new_wvl = []
    new_flux = []
    new_uncer = []
    new_bin_center = wvl_log[i] + bin_radius_diff + remove_bin_fraction * old_bin_length
    new_wvl.append(np.exp(new_bin_center))
    #print("unpb:", unpairedbins, "remove_bins:", remove_bins, "N1_skip:", N1_skip, "n_new:", n_new, "n_new_2:", n_new_2,               #
     #     "SUM:", n_new+n_new_2, "i:", i, "N1_new:", N1_new, "i + N1_new:", i+N1_new, "BINS:" "brdiff:", brdiff, "wvl[-1]:", wvl[-1], "LAST BIN:", wvl[-1] - brdiff)     
    new_flux.append(n_new * flux[i] + flux[i + 1:i + N1_new].sum() + n_new_2 * flux[i + N1_new]) 
    new_uncer.append(np.sqrt(np.square(n_new * uncer[i]) + np.sum(np.square(uncer[i + 1:i + N1_new])) + np.square(n_new_2 * uncer[i + N1_new])))
    i = i + N1_new - N1
    while i + N1 + (int(N - 1.0000000000001 + n_new_2) + 1) <= len(wvl_log) - 1:
        #print(i + N1 + (int(N - 1.0000000000001 + n_new_2) + 1), "i:", i, "N3:", N3, "N1:", N1, 
        #      "n_new_0:", n_new_0, "n_new_2:", n_new_2, "i + N1:", i+N1, "n_test:", n_test, "N1_new:", N1_new)
        #print("OLD =>" "n_new_2:", n_new_2, "n_new_0:", n_new_0, "i:", i, "i + N1:", i + N1 + (int(N - 1 + n_new_2) + 1))
        i = i + N1
        k = k + N3
        N1 = int(N - 1.0000000000001 + n_new_2) + 1
        N3 = int(N - 1.0000000000001 - n_new_0) + 1
        n_new_0 = N - N1 + n_new_0
        #print("n_new_2:",1 - n_new_2, "N_NEW_0:", n_new_0, "i:", i, "i + N1:", i + N1, "N1:", N1)
        new_bin_center +=  (1 - n_new_2) * (wvl_log[i+1] - wvl_log[i])  + wvl_log[i + N1] - wvl_log[i + 1] + n_new_0 * (wvl_log[i + N1] - wvl_log[i + N1 - 1])
        new_wvl.append(np.exp(new_bin_center))
        new_flux.append((1 - n_new_2) * flux[i] + flux[i + 1:i + N1].sum() + n_new_0 * flux[i + N1])
        new_uncer.append(np.sqrt(np.square((1 - n_new_2) * uncer[i]) + np.sum(np.square(uncer[i + 1:i + N1])) + np.square(n_new_0 * uncer[i + N1])))
        n_new_2 = N - N1 + n_new_2
        #print("SHAPE:", "SHOULD:", int(len(wvl)/N))
    return np.array(new_wvl), np.array(new_flux), np.array(new_uncer)


# Drops bins from left.

def mybinspec(wvl, flux, uncer, bin_length):
    data_dtype = flux.dtype
    flux = flux.astype('float64')
    phase_key = list(uncer.keys())[0]
    uncer = uncer[phase_key]
    owbin = (np.log(wvl[1]) - np.log(wvl[0]))
    bin_factor = float(bin_length) / owbin
    #bin_factor = float(wbin) / owbin
    wvl_log = np.log(wvl)
    number_of_newbins = float(len(wvl_log)) / bin_factor
    if float(number_of_newbins).is_integer():
        N = bin_factor - 0.000000000000001
    else:
        N = bin_factor
    N1 = int(N)
    N3 = int(N)
    #######################################################################################################
    integer_number_of_new_bins = int(len(wvl_log) / N)                                                    #
    used_bins = integer_number_of_new_bins * N                                                            #
    unpaired_bins = len(wvl_log) - used_bins                                                              #
    N1_skip = int(unpaired_bins)                                                                          #
    remove_bin_fraction = unpaired_bins - int(unpaired_bins)                                              #
    n_new = 1 - remove_bin_fraction                                                                       #
    n_new_rem = N - n_new                                                                                 #
    n_new_2 = n_new_rem - int(n_new_rem - 0.00000000000001)                                               #
    n_new_0 = n_new_rem - int(n_new_rem - 0.00000000000001)                                               #
    N1_new = int(np.ceil(n_new_rem - 0.00000000000001))                                                   #
    #######################################################################################################
    i = N1_skip
    k = N1_skip
    old_bin_length = (wvl_log[i + 1] - wvl_log[i])              
    old_bin_radius = float(old_bin_length) / 2.0                 
    new_bin_length = old_bin_length * N                          
    new_bin_radius = float(new_bin_length) / 2.0                 
    bin_radius_diff = new_bin_radius - old_bin_radius
    new_wvl = []
    new_flux = []
    new_uncer = []
    new_bin_center = wvl_log[i] + bin_radius_diff + remove_bin_fraction * old_bin_length
    new_wvl.append(np.exp(new_bin_center))
    new_flux.append(n_new * flux[i] + flux[i + 1:i + N1_new].sum() + n_new_2 * flux[i + N1_new]) 
    new_uncer.append(np.sqrt(np.square(n_new * uncer[i]) + np.sum(np.square(uncer[i + 1:i + N1_new])) + np.square(n_new_2 * uncer[i + N1_new])))
    i = i + N1_new - N1
    while i + N1 + (int(N - 1.0000000000001 + n_new_2) + 1) <= len(wvl_log) - 1:
        i = i + N1
        k = k + N3
        N1 = int(N - 1.0000000000001 + n_new_2) + 1
        N3 = int(N - 1.0000000000001 - n_new_0) + 1
        n_new_0 = N - N1 + n_new_0
        new_bin_center +=  (1 - n_new_2) * (wvl_log[i+1] - wvl_log[i])  + wvl_log[i + N1] - wvl_log[i + 1] + n_new_0 * (wvl_log[i + N1] - wvl_log[i + N1 - 1])
        new_wvl.append(np.exp(new_bin_center))
        new_flux.append((1 - n_new_2) * flux[i] + flux[i + 1:i + N1].sum() + n_new_0 * flux[i + N1])
        new_uncer.append(np.sqrt(np.square((1 - n_new_2) * uncer[i]) + np.sum(np.square(uncer[i + 1:i + N1])) + np.square(n_new_0 * uncer[i + N1])))
        n_new_2 = N - N1 + n_new_2
    return np.array(new_wvl), np.array(new_flux), np.array(new_uncer)


def lowres_dataset(dataset, wbin):
    dataset_value_list = list(dataset.values())
    dataset_key_list = list(dataset.keys())
    lowres_dataset_dict = {}
    #bin_factor = rebin_spec(wbin)
    sn = 0
    while sn < len(dataset_value_list):
        dataset_value_list[sn].rebin_spec(wbin)
        lowres_dataset_dict[dataset_key_list[sn]] = dataset_value_list[sn]
        lowres_dataset_dict.update(lowres_dataset_dict)
        sn = sn + 1
    return lowres_dataset_dict#, bin_factor



def random_noise_dataset(dataset):
    dataset_value_list = list(dataset.values())
    dataset_key_list = list(dataset.keys())
    random_dataset_dict = {}
    #bin_factor = rebin_spec(wbin)
    sn = 0
    while sn < len(dataset_value_list):
        dataset_value_list[sn].random_noise_spec()
        random_dataset_dict[dataset_key_list[sn]] = dataset_value_list[sn]
        random_dataset_dict.update(random_dataset_dict)
        sn = sn + 1
    return random_dataset_dict



def GetTypes(dataset):
    keys = list(dataset.keys())
    type_list = []
    type_dict = {}
    for i in range(0, len(keys)):
        types = dataset[keys[i]].header['TypeStr']
        type_list.append(types)
        type_dict[keys[i]] = types
    return type_dict


def ArangeSNeType(dataset):
    keys = list(dataset.keys())
    type_Ib_norm_list = []
    type_Ib_pec_list = []
    type_IIb_list = []
    type_Ic_list = []
    type_Ic_norm_list = []
    type_Ic_pec_list = []
    type_Ic_broad_list = []
    type_dict = {}
    for i in range(0, len(keys)):
        if dataset[keys[i]].header['TypeStr'] == 'Ib-norm':
            type_Ib_norm_list.append(keys[i])
        elif dataset[keys[i]].header['TypeStr'] == 'Ib-pec':
            type_Ib_pec_list.append(keys[i])
        elif dataset[keys[i]].header['TypeStr'] == 'IIb':
            type_IIb_list.append(keys[i])
        elif dataset[keys[i]].header['TypeStr'] == 'Ic':
            type_Ic_list.append(keys[i])
        elif dataset[keys[i]].header['TypeStr'] == 'Ic-norm':
            type_Ic_norm_list.append(keys[i])
        elif dataset[keys[i]].header['TypeStr'] == 'Ic-pec':
            type_Ic_pec_list.append(keys[i])
        elif dataset[keys[i]].header['TypeStr'] == 'Ic-broad':
            type_Ic_broad_list.append(keys[i])
    type_dict['Ib-norm'] = type_Ib_norm_list
    type_dict['Ib-pec'] = type_Ib_pec_list
    type_dict['IIb'] = type_IIb_list
    type_dict['Ic'] = type_Ic_list
    type_dict['Ic-norm'] = type_Ic_norm_list
    type_dict['Ic-pec'] = type_Ic_pec_list
    type_dict['Ic-broad'] = type_Ic_broad_list
    return type_dict