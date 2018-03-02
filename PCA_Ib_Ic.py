
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import sys, os, glob, re
import sklearn
import plotly.plotly as ply
import plotly.graph_objs as go
import plotly.tools as tls

from scipy.interpolate import interp1d


from config import *

        #Normalize an NxM matrix, where N is the number of spectra, 
        #and M is the dimension of each wavelength (ie number of wavelength bins)

def normalize(spec_mat):
    for i in range(spec_mat.shape[0]):
        spec_mat[i][spec_mat[i]==0] = np.nan
        
    spectra_mean = np.nanmean(spec_mat, axis=1)
    spectra_std = np.nanstd(spec_mat, axis=1)
    spec_matT = (spec_mat.T - spectra_mean)/spectra_std
    spec_mat = spec_matT.T
    return spec_mat


def rebin(spec_mat, smoothing, wvl=None):

    nrows, ncols = spec_mat.shape
    tmp = np.reshape(spec_mat,(nrows, ncols/smoothing, smoothing))
    spec_mat = np.nanmean(tmp, axis=2)
    print spec_mat.shape
    
    if wvl!=None:
        wvrows = wvl.shape[0]
        wvcols = 1
        wvtemp = np.reshape(wvl, (wvrows/smoothing,1,smoothing))
        print wvtemp.shape
        wvl = np.nanmean(wvtemp, axis=2)
        print wvl.shape
    if wvl!=None:
        return spec_mat, wvl
    else:
        return spec_mat

    
def percent_reconstruct(eig_vals, ncomp):
    abs_eig = np.fabs(eig_vals)
    norm = np.sum(abs_eig)
    percent = abs_eig / norm
    return np.sum(percent[:ncomp])

def plotly_corner_spec(n):
    sp=[]
    for i in range(1,n):
        l = i*[{}]
        for i in range(n-i-1):
            l.append(None)
        sp.append(l)
    return sp

if __name__ == '__main__':
    # # Get Data

    print ("looking for spectra in ", snid_dir)
    os.chdir(snid_dir)
    all_spec = glob.glob('*.lnw')
    os.chdir('..')

    # Get List of spectra files that are type Ib or Ic

    # For some reason, these files do not have a standard organization, which makes it
    # stupidly aggravating to load them all. Here we search for spectra of type Ib or Ic
    # and find the line number where the wavelength 2501.69 appears (this is the first).
    Ib_Ic = []
    Ib_Ic_names = []
    skiprows = []
    SN_type_vec = []
    phase_type = []
    phases = []
    for spec_name in all_spec:
        path = snid_dir + spec_name
        with open(path, 'r') as f:
            lines = f.readlines()
            header = lines[0].split()
            SN_type = int(header[-2])
            SN_sub_type = int(header[-1])
            if ((SN_type == 2) or (SN_type == 3)):
            
                Ib_Ic.append(path)
                Ib_Ic_names.append(spec_name[:-4])
                type_tup = [SN_type, SN_sub_type]
                SN_type_vec.append(type_tup)
                #the first observed wavelength is always 2501.69
                first_wvl = '2501.69'
                for i in range(len(lines)-1):
                    line = lines[i]
                    if first_wvl in line:
                        skiprows.append(i)
                        phase_row = lines[i-1].split()
                        phase_type.append(int(phase_row[0]))
                        phases.append([float(ph) for ph in phase_row[1:]])
                        break
                    
    phase_type = np.array(phase_type)
    phases = [np.array(ph) for ph in phases]
    SN_type_vec = np.array(SN_type_vec)
    Ib_Ic_names = np.array(Ib_Ic_names)

    # phase value is the phase, using the definition where phase is number
    # of days after max light. This code finds the observed phase closest to 
    # the chose phase_value.

    phase_value = 0
    phase_cols = []
    phase_arr = []
    for i in range(len(phase_type)):
        idx = (np.abs(phases[i] - phase_value)).argmin()
        phase_cols.append(idx)
        phase_arr.append(phases[i][idx])
    print phase_cols
    phase_arr = np.array(phase_arr)
    #phase_arr[3]=1
    #print phase_arr[3]
    
    plt.hist(phase_arr[phase_type==0][phase_arr[phase_type==0]>-500])

    print "There are %d type Ib and Ic SN"%(len(Ib_Ic))

    # For each spectra, skip all the header lines about removing continuum and load the phase=0
    # spectra into a numpy array. Do some data processing like filtering for desired wavelengths and 
    # normalizing data.

    min_wavelength = 4000
    max_wavelength = 7000

    spectra = []
    for i in range(len(Ib_Ic)):
        spec = Ib_Ic[i]
        skiprow = skiprows[i]
        s = np.loadtxt(spec,skiprows=skiprow, usecols=(0,phase_cols[i]+1)) # Note the +1 because in the SNID files there is a 0/1 phase type
    
        # check that first wvl is 2501.69
        if not(s[0][0] == 2501.69): print 'first wavelength is not 2501.69'
    
        # filter for desired wavelengths
        mask = np.logical_and(s[:,0] > min_wavelength, s[:,0] < max_wavelength)
        s = s[mask]
        spectra.append(s)
    
    SN_type_vec = np.array(SN_type_vec)

    # # Load all phases for specific SN
    #snid_list = ['sn2009jf', 'sn2007gr', 'sn2004gq']
    snid_list = allIc # All Ic SNe

    snid_list = allIb  # All Ib SNe


    snid_s_list = []
    snid_phase_list = []


    for snid in snid_list:
        #snid = 'sn2009jf'
        snid_ind = np.where(Ib_Ic_names==snid)[0][0]
        snid_spec = Ib_Ic[snid_ind]
        snid_skiprows = skiprows[snid_ind]
        snid_s = np.loadtxt(snid_spec, skiprows=snid_skiprows)
        snid_s = snid_s[np.logical_and(snid_s[:,0]<7000,snid_s[:,0]>4000)]
        snid_s = snid_s[:,1:]
        snid_s = snid_s.T
        snid_phases = phases[snid_ind]
    
        snid_s_list.append(snid_s)
        snid_phase_list.append(snid_phases)
    

    
    for i, snid_ph in enumerate(snid_phase_list):
        print snid_list[i]
        print snid_ph


        # ----------------------------------------




    #use09jf = [-18, -15, -9, -5, 0, 7, 11, 18, 30, 46.6, 95.2]
    #use07gr = [-9, -5, 0, 5, 10, 15, 20, 25, 42, 45, 49]
    #use04gq = [-9, -5.1, -1.1, -0.1, 15.8, 17.8, 24.7, 41.6, 54.6]
    #use = [use09jf, use07gr, use04gq]
    
    #phase_mask = np.array([ph in use for ph in snid_phases])

    for i,snid_s in enumerate(snid_s_list):
        phase_mask = np.array([ph in use[i] for ph in snid_phase_list[i]])
        snid_s_list[i] = snid_s[phase_mask]
        snid_phases = snid_phase_list[i]
        snid_phase_list[i] = snid_phases[phase_mask]
    print snid_phase_list
    for snid_s in snid_s_list:
        print ("snid_s shape", snid_s.shape)



    for i,snid_s in enumerate(snid_s_list):
        snid_s_list[i] = normalize(snid_s)

    for i,snid_s in enumerate(snid_s_list):
        snid_s_list[i] = rebin(snid_s, smoothing=2)

    for snid_s in snid_s_list: print snid_s.shape


    for i,snid_s in enumerate(snid_s_list):

        big_gap_mask = np.array([np.isnan(spec).any() for spec in snid_s])
        print 'total number of bad spectra = %d'%(np.sum(big_gap_mask))
        snid_s_list[i] = snid_s[np.invert(big_gap_mask)]
        snid_phases = snid_phase_list[i]
        snid_phase_list[i] = snid_phases[np.invert(big_gap_mask)]


    snid_s = snid_s[np.invert(big_gap_mask)]
    snid_phases = snid_phases[np.invert(big_gap_mask)]


    for snid_s in snid_s_list: print snid_s.shape
    for snid_ph in snid_phase_list: print snid_ph.shape
    

    snid_s_list

    # Organize spectra in an (N x M) array, where N is the number of spectra, and M is the dimensionality
    # (ie number of data points in spectrum) of a spectrum. We treat each wavelength as independent
    # and zero the means and unify the standard deviations.

    wavelengths = np.array(spectra[0][:,0])
    spectra_matrix = np.ndarray((len(spectra),spectra[0].shape[0]))
    for i, spec in enumerate(spectra):
        spectra_matrix[i,:] = spec[:,1]
    
    #spectra_mean = np.mean(spectra_matrix, axis=1)
    #spectra_std = np.std(spectra_matrix, axis=1)

    phase_range = np.logical_and(phase_arr>-5, phase_arr<5)
    phase_mask = np.logical_and(phase_type==0, phase_range)
    #phase_mask = phase_type==0
    print len(phase_mask)
    print spectra_matrix.shape

    phase_cols = np.array(phase_cols)
    phase_cols = phase_cols[phase_mask]
    skiprows = np.array(skiprows)
    skiprows = skiprows[phase_mask]
    
    spectra_matrix[phase_mask].shape
    spectra_matrix = spectra_matrix[phase_mask]
    SN_type_vec = SN_type_vec[phase_mask]
    phase_arr = phase_arr[phase_mask]
    Ib_Ic_names = Ib_Ic_names[phase_mask]
    print SN_type_vec.shape
    print spectra_matrix.shape
    print Ib_Ic_names.shape

    IcBLmask = np.array([np.array_equal(arr, np.array([3,4])) for arr in SN_type_vec])
    

    SN_type_vec[IcBLmask]

    import pidly
    idl = pidly.IDL()
    idl('.COMPILE type default lmfit linear_fit powerlaw_fit integ binspec SNspecFFTsmooth')

    spectra_matrix[IcBLmask].shape

    IcBL_smoothed_matrix = []
    IcBL_pre_smooth = []
    IcBL_pre_smooth_IDL = []
    
    for i in range(len(IcBLmask[IcBLmask==True])):
        spec_name = Ib_Ic_names[IcBLmask][i]
        print spec_name
        spec_path = './allSNIDtemp/'+spec_name+'.lnw'
        skiprow = skiprows[IcBLmask][i]
        usecol = phase_cols[IcBLmask][i]+1
        s = np.loadtxt(spec_path,skiprows=skiprow, usecols=(0,usecol))
    
        min_wavelength = 4000
        max_wavelength = 7000
        #mask = np.logical_and(s[:,0]>min_wavelength, s[:,0]<max_wavelength)
        #s = s[mask]
        if i==0:
            IcBL_wvl = s[:,0]
        with open('tmp_spec.txt','w') as f:
            for j in range(s.shape[0]):
                f.write('        %.4f        %.7f\n'%(s[j,0],s[j,1]+10.0))
            f.close()
    
        idl('readcol, "tmp_spec.txt", w, f')
        idl('SNspecFFTsmooth, w, f, 3000, f_ft, f_std, sep_vel')
        IcBL_pre_smooth.append(s[:,1])
        IcBL_smoothed_matrix.append(idl.f_ft)
        IcBL_pre_smooth_IDL.append(idl.f)

    IcBL_pre_smooth = np.array(IcBL_pre_smooth)
    IcBL_smoothed_matrix = np.array(IcBL_smoothed_matrix)
    IcBL_pre_smooth_IDL = np.array(IcBL_pre_smooth_IDL)

    plt.plot(IcBL_wvl,IcBL_pre_smooth[14],color='b')
    #plt.plot(IcBL_wvl,IcBL_smoothed_matrix[14],color='r')
    #plt.plot(IcBL_wvl,IcBL_pre_smooth_IDL[14],color='g')

    plt.plot(IcBL_wvl, (IcBL_smoothed_matrix[14]-np.mean(IcBL_smoothed_matrix))/np.std(IcBL_smoothed_matrix), color='b')
    plt.plot(IcBL_wvl, (IcBL_pre_smooth[14]-np.mean(IcBL_pre_smooth[14]))/np.std(IcBL_pre_smooth[14]), color='r')


    smooth_mean = np.nanmean(IcBL_smoothed_matrix, axis=1)

    smooth_std = np.nanstd(IcBL_smoothed_matrix, axis=1)
    original_smooth_spectra = np.copy(IcBL_smoothed_matrix)
    IcBL_smoothed_matrixT = (IcBL_smoothed_matrix.T - smooth_mean)/smooth_std
    IcBL_smoothed_matrix = IcBL_smoothed_matrixT.T

    pre_smooth_mean = np.nanmean(IcBL_pre_smooth, axis=1)
    pre_smooth_std = np.nanstd(IcBL_pre_smooth, axis=1)
    original_pre_smooth_spectra = np.copy(IcBL_pre_smooth)
    IcBL_pre_smooth_matrixT = (IcBL_pre_smooth.T - pre_smooth_mean)/pre_smooth_std
    IcBL_pre_smooth = IcBL_pre_smooth_matrixT.T

    plt.plot(IcBL_wvl,original_smooth_spectra[1])




    f = plt.figure(figsize=(15,50))
    nshow=IcBL_smoothed_matrix.shape[0]
    for i in range(nshow):
        plt.subplot(nshow,1,i+1)
        name = Ib_Ic_names[IcBLmask][i]
        plt.plot(IcBL_wvl, IcBL_smoothed_matrix[i], label="smoothed "+name,color='b')
        #plt.plot(IcBL_wvl, -IcBL_smoothed_matrix[i], label="smoothed "+name,color='g')
        plt.plot(IcBL_wvl, IcBL_pre_smooth[i], label="pre_smoothed "+name,color='r')
        if i==0:
            plt.title('Smoothed IcBL spectra 15$\pm$5')
        if i==nshow-1:
            plt.xlabel('Wavelength')
        plt.ylabel('Rel Flux')
        plt.legend(fontsize=12)
        
    phase_cols = np.array(phase_cols)
    phase_cols[IcBLmask]

    skiprows.shape
    
    IcBLmask.shape

    plt.hist(phase_arr)
    plt.show()

    for i in range(spectra_matrix.shape[0]):
        spectra_matrix[i][spectra_matrix[i]==0] = np.nan

    spectra_mean = np.nanmean(spectra_matrix, axis=1)
    spectra_std = np.nanstd(spectra_matrix, axis=1)
    original_spectra = np.copy(spectra_matrix)
    spectra_matrixT = (spectra_matrix.T - spectra_mean)/spectra_std
    spectra_matrix = spectra_matrixT.T



    for i,sn in enumerate(Ib_Ic_names[IcBLmask]):
        print sn
        smooth_spec = IcBL_smoothed_matrix[i]
        
        min_wavelength = 4000
        max_wavelength = 7000
        mask = np.logical_and(IcBL_wvl>min_wavelength, IcBL_wvl<max_wavelength)
        smooth_spec = smooth_spec[mask]
        #print smooth_spec
        ind = np.where(Ib_Ic_names==sn)[0][0]
        spectra_matrix[ind] = smooth_spec

    print wavelengths.shape
    print spectra_matrix[Ib_Ic_names=='sn2007ru'].shape
    plt.plot(wavelengths, spectra_matrix[Ib_Ic_names=='sn2007ru'][0])

    plt.figure(figsize=(15,25))
    for i,spec in enumerate(spectra_matrix):
        alpha=0.25
        if not i%10:
            alpha=1
        plt.plot(wavelengths,spec+i*2, alpha=alpha)

    plt.show()

    test_spec = spectra_matrix[0]
    plt.figure(figsize=(10,5))
    plt.plot(wavelengths,test_spec,'c')
    plt.show()

    good_values = np.isfinite(test_spec)
    interp_fn = interp1d(wavelengths[good_values],test_spec[good_values])

    plt.figure(figsize=(10,5))
    p1 = plt.plot(wavelengths,interp_fn(wavelengths),'r',label="interpolated")
    p2 = plt.plot(wavelengths,test_spec,'c',label="real data")
    plt.legend()
    plt.shape()

    print spectra_matrix.shape
    print wavelengths.shape

    nrows, ncols = spectra_matrix.shape
    smoothing = 2 # Width of bins in terms of number of wavelength steps
    tmp = np.reshape(spectra_matrix,(nrows, ncols/smoothing, smoothing))

    spectra_matrix = np.nanmean(tmp, axis=2)

    spectra_matrix.shape


    wvrows = wavelengths.shape[0]
    wvcols = 1
    wvtemp = np.reshape(wavelengths, (wvrows/smoothing,1,smoothing))
    print wvtemp.shape
    wavelengths = np.nanmean(wvtemp, axis=2)

    wavelengths.shape

    a = np.ones((3,4))
    a - np.array([0,1,2,1])
    

    sample_mean = np.nanmean(spectra_matrix,axis=0)
    #spectra_matrix = spectra_matrix - sample_mean

    # Plot the spectra post data processing
    
    plt.figure(figsize=(15,25))
    for i,spec in enumerate(spectra_matrix):
        alpha=0.25
        if not i%10:
            alpha=1
        plt.plot(wavelengths,spec+i*2, alpha=alpha)

    big_gap_mask = np.array([np.isnan(spec).any() for spec in spectra_matrix])
    print 'total number of bad spectra = %d'%(np.sum(big_gap_mask))


    spectra_matrix = spectra_matrix[np.invert(big_gap_mask)]
    SN_type_vec = SN_type_vec[np.invert(big_gap_mask)]
    Ib_Ic_names = Ib_Ic_names[np.invert(big_gap_mask)]

    phase_arr = phase_arr[np.invert(big_gap_mask)]

    

    phase_arr.shape


    print spectra_matrix.shape
    print SN_type_vec.shape
    print Ib_Ic_names.shape

    data = []
    if PLOTLY:
        for i,spec in enumerate(spectra_matrix):
            alpha=0.25
            if not i%10:
                alpha=1
            trace = go.Scatter(x=wavelengths, y=spec+i*2, mode='lines',
                           text=len(wavelengths)*[Ib_Ic_names[i]])
            data.append(trace)
        layout = go.Layout(showlegend=False, autosize=False, width=1000,
                           height=2000, title='SNID Spectra',
                           xaxis=dict(title='Wavelength'))
        fig = go.Figure(data=data, layout=layout)
        ply.iplot(fig)
        #plt.plot(wavelengths,spec+i*2, alpha=alpha)



    print (PCAdecompspec(Ib_Ic_names, spectra_matrix, SN_type_vec,
                  phase_arr, 8).shape)
    


