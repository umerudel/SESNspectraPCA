# Class written to run PCA analysis on SNID supernova spectra.
# Author: Marc Williamson
# Date created: 3/01/2018
from __future__ import division
#import matplotlib
#matplotlib.use('Agg')
import numpy as np
import scipy
from scipy.interpolate import interp1d
import scipy.optimize as opt
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_color_codes('colorblind')
import sys, os, glob, re, copy
import sklearn
from sklearn.decomposition import PCA
import plotly.plotly as ply
import plotly.graph_objs as go
import plotly.tools as tls
import pickle
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from scipy.spatial import distance

try:
    import pidly
except ImportError:
    print 'You do not have pidly installed! Install it with pip, or from https://github.com/anthonyjsmith/pIDLy'
    print 'In order to use smoothed IcBL spectra, you need pidly and SNspecFFTsmooth.pro from https://github.com/nyusngroup/SESNspectraLib'
    print 'You can also use the example pickled object provided that has had IcBL smoothed already.'

# Global Variables
firstSNIDWavelength = '2501.69' # the first observed wavelength in all the SNID files


# Private helper functions



# The following function finds the observed phase closest to the loadPhase value
# specified by the user for each SNe. It returns phaseCols, a list of the column
# number in the SNID file of the desired phase (NEED to add 1 because there is a 
# Wavelength column in the SNID file). It also returns phaseArr, an array of the 
# closest phase for each SNe in the sample.
def findClosestObsPhase(phases, loadPhase):        
    phaseCols = []
    phaseArr = []
    for i in range(len(phases)):
        idx = (np.abs(phases[i] - loadPhase)).argmin()
        phaseCols.append(idx)
        phaseArr.append(phases[i][idx])
    phaseArr = np.array(phaseArr)
    phaseCols = np.array(phaseCols)
    return phaseCols, phaseArr



# Function to get masks of IIb, Ib, Ic, and IcBL SNe. These masks are used internally
# for plotting the different SNe types in different colors.
def getSNeTypeMasks(sneTypes):
    IIbMask = np.array([np.array_equal(typeArr,[2,4]) for typeArr in sneTypes])
    
    tmp1 = np.array([np.array_equal(typeArr,[2,1]) for typeArr in sneTypes])
    tmp2 = np.logical_or(tmp1, np.array([np.array_equal(typeArr,[2,2]) for typeArr in sneTypes]))
    tmp3 = np.logical_or(tmp2, np.array([np.array_equal(typeArr,[2,3]) for typeArr in sneTypes]))
    IbMask = np.logical_or(tmp3, np.array([np.array_equal(typeArr,[2,5]) for typeArr in sneTypes]))
    
    IcBLMask = np.array([np.array_equal(typeArr,[3,4]) for typeArr in sneTypes])

    tmp1 = np.array([np.array_equal(typeArr,[3,1]) for typeArr in sneTypes])
    tmp2 = np.logical_or(tmp1, np.array([np.array_equal(typeArr,[3,2]) for typeArr in sneTypes]))
    IcMask = np.logical_or(tmp2, np.array([np.array_equal(typeArr,[3,3]) for typeArr in sneTypes]))

    return IIbMask, IbMask, IcMask, IcBLMask

# Binspec implemented in python.
def binspec(wvl, flux, wstart, wend, wbin):
    nlam = (wend - wstart) / wbin + 1
    nlam = int(np.ceil(nlam))
    outlam = np.arange(nlam) * wbin + wstart
    answer = np.zeros(nlam) 
    interplam = np.unique(np.concatenate((wvl, outlam)))  
    interpflux = np.interp(interplam, wvl, flux)
    
    for i in np.arange(0, nlam - 1):
        cond = np.logical_and(interplam >= outlam[i], interplam <= outlam[i+1])
        w = np.where(cond)
        if len(w) == 2:
            answer[i] = 0.5*(np.sum(interpflux[cond])*wbin)
        else:
            answer[i] = scipy.integrate.simps(interpflux[cond], interplam[cond])

    answer[nlam - 1] = answer[nlam - 2]
    cond = np.logical_or(outlam >= max(wvl), outlam < min(wvl))
    answer[cond] = 0
    return answer/wbin, outlam

#smooth spectrum using SNspecFFTsmooth procedure
def smooth(wvl, flux, cut_vel):
    c_kms = 299792.47 # speed of light in km/s
    vel_toolarge = 100000 # km/s
    
    wvl_ln = np.log(wvl)
    num = wvl_ln.shape[0]
    binsize = wvl_ln[-1] - wvl_ln[-2]
    f_bin, wln_bin = binspec(wvl_ln, flux, min(wvl_ln), max(wvl_ln), binsize)
    fbin_ft = np.fft.fft(f_bin)#*len(f_bin)
    freq = np.fft.fftfreq(num)
    num_upper = np.max(np.where(1.0/freq[1:] * c_kms * binsize > cut_vel))
    num_lower = np.max(np.where(1.0/freq[1:] * c_kms * binsize > vel_toolarge))
    mag_avg = np.mean(np.abs(fbin_ft[num_lower:num_upper+1]))
    powerlaw = lambda x, amp, exp: amp*x**exp
    
    #do linear regression on log data to obtain a guess for powerlaw parameters
    
    print num_upper
#    print nup
    num_bin = len(f_bin)
    xdat = freq[num_lower:num_upper]
#    print xdat
    ydat = np.abs(fbin_ft[num_lower:num_upper])
#    print ydat
    nonzero_mask = xdat!=0            
    slope, intercept, _,_,_ = st.linregress(np.log(xdat[nonzero_mask]), np.log(ydat[nonzero_mask]))
    exp_guess = slope
    amp_guess = np.exp(intercept)
#    print slope, intercept
    
    #do powerlaw fit
    xdat = freq[num_lower:int(num_bin/2)]
    ydat = np.abs(fbin_ft[num_lower:int(num_bin/2)])
    #exclude data where x=0 because this can cause 1/0 errors if exp < 0
    finite_mask = np.logical_not(xdat==0)
    finite_mask = np.logical_and(finite_mask, np.isfinite(ydat))
    ampfit, expfit = opt.curve_fit(powerlaw, xdat[finite_mask], ydat[finite_mask], p0=[amp_guess, exp_guess])[0]

    #find intersection of average fbin_ft magnitude and powerlaw fit to calculate separation
    #velocity between signal and noise.
    intersect_x = np.power((mag_avg/ampfit), 1.0/expfit)
    sep_vel = 1.0/intersect_x * c_kms * binsize
    
    #filter out frequencies with velocities higher than sep_vel
    smooth_fbin_ft = np.array([fbin_ft[ind] if np.abs(freq[ind])<np.abs(intersect_x) else 0 \
                               for ind in range(len(freq))])#/len(f_bin)
    #inverse fft on smoothed fluxes
    smooth_fbin_ft_inv = np.real(np.fft.ifft(smooth_fbin_ft))
    
    #interpolate smoothed fluxes back onto original wavelengths
    w_smoothed = np.exp(wln_bin)
    f_smoothed = np.interp(wvl, w_smoothed, smooth_fbin_ft_inv)
    
    return w_smoothed, f_smoothed, sep_vel



def findGaps(specAllWvl, wavelengths, minwvl, maxwvl):
    wvlMask = np.logical_and(wavelengths > minwvl, wavelengths < maxwvl)
    spec = specAllWvl[wvlMask]
    longestGap = 0
    endGap = False
    startGap = False
    whereNan = np.where(np.isnan(spec))[0]
    if len(whereNan) == 0:
        return 0, False, False
    if whereNan[0] == 0:
        startGap = True
    if whereNan[-1] == len(spec)-1:
        endGap = True

    gapL = -1
    for i, ind in enumerate(whereNan):
        if i==0:
            gapL = 1
        else:
            oldInd = whereNan[i-1]
            if oldInd == ind-1:
                gapL = gapL +1
            else:
                if gapL > longestGap:
                    longestGap = gapL
                gapL = 1

    return longestGap, startGap, endGap

def findGapLength(wvl, wavelengths, specAllWvl, direction):
    wvlind = np.where(wavelengths == wvl)[0][0]
    fluxfinite = np.where(np.isfinite(specAllWvl))[0]
    if wvlind in fluxfinite:
        return 0, wvl
    gapEndIndArr = np.where(fluxfinite > wvlind)[0]
    if len(gapEndIndArr) == 0:
        gapEndInd = len(wavelengths)
    else:
        gapEndInd = fluxfinite[gapEndIndArr[0]]

    gapStartIndArr = np.where(fluxfinite < wvlind)[0]
    if len(gapStartIndArr) == 0:
        gapStartInd = -1
    else:
        gapStartInd = fluxfinite[gapStartIndArr[-1]]
    gapLength = gapEndInd - gapStartInd + 1 - 2

    if direction=='left':
        if gapStartInd == -1:
            returnWvl = -1
        else:
            returnWvl = wavelengths[gapStartInd]
    if direction=='right':
        if gapEndInd == len(wavelengths):
            returnWvl = -1
        else:
            returnWvl = wavelengths[gapEndInd]
    return gapLength, returnWvl


class NovaPCA:

# Initialize the NovaPCA object with a path to the directory containing all 
# the SNID spectra.
    def __init__(self, snidDirPath):
        self.spectraMatrix = None
        self.pcaCoeffMatrix = None
        self.obsSNIDPhases = None
        self.sneNames = None
        self.sneTypes = None
        self.skiprows = None
        self.phaseCols = None
        self.spectraMean = None
        self.spectraStd = None
        
        self.IIb_color = 'g'
        self.Ib_color = 'mediumorchid'
        self.Ic_color = 'r'
        self.IcBL_color = 'k'
        self.H_color = 'steelblue'
        self.He_color = 'indianred'


        self.maskDocs = []
        self.maskAttributes = ['spectraMatrix','pcaCoeffMatrix','obsSNIDPhases','sneNames','sneTypes',\
                              'skiprows', 'phaseCols', 'spectraMean', 'spectraStd']

        self.sklearnPCA = None
        self.evecs = None
        self.evals = None
        self.evals_cs = None

        self.wavelengths = None
        self.loadPhase = None
        self.phaseWidth = None
        self.maskList = []
        self.snidDirPath = snidDirPath
        
        return

# Method to load SNID spectra of the types specified in loadTypes. 
# The SNID type structure is listed at the end of this file.
# Arguments:
#     loadTypes -- list of 2tuples specifying which SNe types to load.
#                  See SNID structure at end of file.
#                  Ex) >>> loadTypes = [(1,2), (2,2), (3,2)] loads Ia-norm, Ib-norm, and Ic-norm.
#     phaseType -- int, either 0 for phases measured relative to max light, or 1 
#                  for phases measured relative to first observation.
#     loadPhase -- float, what phase you want to load the nearest observed spectra of for each SNe.
#     loadPhaseRangeWidth -- float, width of phase range you want to allow, ie phase = 15 +/- 5 days.
#     minwvl/maxwvl -- wavelength cutoffs for loading the spectra.

    def loadSNID(self, loadTypes, phaseType, loadPhase, loadPhaseRangeWidth, minwvl, maxwvl, gap):
        """
 Method to load SNID spectra of the types specified in loadTypes. 
 Arguments:
     loadTypes -- list of 2tuples specifying which SNe types to load.
                  See SNID structure at end of this docstring.
                  Ex) >>> loadTypes = [(1,2), (2,2), (3,2)] loads Ia-norm, Ib-norm, and Ic-norm.
     phaseType -- int, either 0 for phases measured relative to max light, or 1 
                  for phases measured relative to first observation.
     loadPhase -- float, what phase you want to load the nearest observed spectra of for each SNe.
     loadPhaseRangeWidth -- float, width of phase range you want to allow, ie phase = 15 +/- 5 days.
     minwvl/maxwvl -- wavelength cutoffs for loading the spectra.

SNID Type Structure:

#* SN Ia
#      typename(1,1) = 'Ia'      ! first element is name of type
#      typename(1,2) = 'Ia-norm' ! subtypes follow...(normal, peculiar, etc.)
#      typename(1,3) = 'Ia-91T'
#      typename(1,4) = 'Ia-91bg'
#      typename(1,5) = 'Ia-csm'
#      typename(1,6) = 'Ia-pec'
#      typename(1,7) = 'Ia-99aa'
#      typename(1,8) = 'Ia-02cx'

#* SN Ib      
#      typename(2,1) = 'Ib'
#      typename(2,2) = 'Ib-norm'
#      typename(2,3) = 'Ib-pec'
#      typename(2,4) = 'IIb'     ! IIb is not included in SNII
#      typename(2,5) = 'Ib-n'    ! Ib-n can be regarded as a kind of Ib-pec 

#* SN Ic
#      typename(3,1) = 'Ic'
#      typename(3,2) = 'Ic-norm'
#      typename(3,3) = 'Ic-pec'
#      typename(3,4) = 'Ic-broad'

#* SN II
#      typename(4,1) = 'II'
#      typename(4,2) = 'IIP'     ! IIP is the "normal" SN II
#      typename(4,3) = 'II-pec'
#      typename(4,4) = 'IIn'
#      typename(4,5) = 'IIL'

#* NotSN
#      typename(5,1) = 'NotSN'
#      typename(5,2) = 'AGN'
#      typename(5,3) = 'Gal'
#      typename(5,4) = 'LBV'
#      typename(5,5) = 'M-star'
#      typename(5,6) = 'QSO'
#      typename(5,7) = 'C-star'
        """
        self.loadPhase = loadPhase
        self.phaseWidth = loadPhaseRangeWidth
        self.minwvl = minwvl
        self.maxwvl = maxwvl

        temp = os.getcwd()
        os.chdir(self.snidDirPath)
        allSpec = glob.glob('*.lnw')
        os.chdir(temp)

        snePaths = []
        sneNames = []
        sneTypeList = []
        skiprows = []
        phaseTypeList = []
        phases = []

        for specName in allSpec:
            path = self.snidDirPath + specName
            with open(path, self.Ic_color) as f:
                lines = f.readlines()
                header = lines[0].split()
                snType = int(header[-2])
                snSubtype = int(header[-1])
                typeTup = (snType, snSubtype)
                if typeTup in loadTypes:
                    snePaths.append(path)
                    sneNames.append(specName[:-4])
                    sneTypeList.append(typeTup)
                    for i in range(len(lines) - 1):
                        line = lines[i]
                        if firstSNIDWavelength in line:
                            skiprows.append(i)
                            phaseRow = lines[i - 1].split()
                            phaseTypeList.append(int(phaseRow[0]))
                            phases.append(np.array([float(ph) for ph in phaseRow[1:]]))
                            break
        phaseTypeList = np.array(phaseTypeList)
        sneTypeList = np.array(sneTypeList)
        sneNames = np.array(sneNames)
        snePaths = np.array(snePaths)
        skiprows = np.array(skiprows)
        print len(phases) #################
        phaseCols, phaseArr = findClosestObsPhase(phases, loadPhase)
        
        spectra = []
        spectraInterp = []
        for i in range(len(snePaths)):
            spec = snePaths[i]
            skiprow = skiprows[i]
            s = np.loadtxt(spec, skiprows=skiprow, usecols=(0,phaseCols[i] + 1)) #Note the +1 becase in the SNID files there is a column (0) for wvl
            mask = np.logical_and(s[:,0] > self.minwvl, s[:,0] < self.maxwvl)
            
            s = s[mask]
            #spectra.append(s[mask])
            spectra.append(s)
            

        wavelengths = np.array(spectra[0][:,0])
        specMat = np.ndarray((len(spectra), spectra[0].shape[0]))
        for i, spec in enumerate(spectra):
            specMat[i,:] = spec[:,1]

        phaseRangeMask = np.logical_and(phaseArr > (loadPhase - loadPhaseRangeWidth), phaseArr < (loadPhase + loadPhaseRangeWidth))
        phaseMask = np.logical_and(phaseTypeList == 0, phaseRangeMask)


        specMat = specMat[phaseMask]
        sneNames = sneNames[phaseMask]
        phaseArr = phaseArr[phaseMask]
        sneTypeList = sneTypeList[phaseMask]
        skiprows = skiprows[phaseMask]
        phaseCols = phaseCols[phaseMask]


        self.sneNames = np.copy(sneNames)
        self.obsSNIDPhases = np.copy(phaseArr)
        self.spectraMatrix = np.copy(specMat)
        self.sneTypes = np.copy(sneTypeList)
        self.wavelengths = np.copy(wavelengths)
        self.skiprows = np.copy(skiprows)
        self.phaseCols = np.copy(phaseCols)

        return



    def smoothSpectra(self, mask, vel_cut, figsize, idlspec=None):
        sepvels = []
        f = plt.figure(figsize=figsize)
        nshow = np.sum(mask)
        plotcounter = 1
        for i in range(len(mask)):
            if mask[i]:
                wsmooth, fsmooth, sepvel = smooth(self.wavelengths, self.spectraMatrix[i], vel_cut[i])
                sepvels.append(sepvel)
                plt.subplot(nshow, 1, plotcounter)
                plotcounter = plotcounter + 1
                name = self.sneNames[i]
                plt.plot(self.wavelengths, self.spectraMatrix[i], label='pre-smoothed '+name, color=self.Ic_color)
                plt.plot(self.wavelengths, fsmooth, label='smoothed sep_vel = '+str(sepvel), color=self.IcBL_color)
                if not idlspec is None:
                    plt.plot(self.wavelengths, idlspec[i], label='idl smoothed', color=self.Ib_color)
                if i==0:
                    plt.title('Smoothed SNe Spectra %d$\pm$%d'%(self.loadPhase, self.phaseWidth))
                if i == nshow - 1:
                    plt.xlabel("Wavelength (Angstroms)")
                plt.ylabel('Rel Flux')
                plt.legend(fontsize=12)
                self.spectraMatrix[i] = fsmooth
        self.smoothSepVels = np.array(sepvels)

        return f




    def spectraGaps(self, maxGapLength):
        gapMask = []
        for spec in self.spectraMatrix:
            spec[spec == 0] = np.nan
            longestGap, startGap, endGap = findGaps(spec, self.wavelengths, self.minwvl, self.maxwvl)
            print startGap, endGap
            if longestGap > maxGapLength:
                gapMask.append(False)
            else:
                startWvl = self.wavelengths[np.where(self.wavelengths > self.minwvl)[0][0]]
                endWvl = self.wavelengths[np.where(self.wavelengths < self.maxwvl)[0][-1]]
                startGapLength, startfiniteWvl = findGapLength(startWvl, self.wavelengths, spec, 'left')
                endGapLength, endfiniteWvl = findGapLength(endWvl, self.wavelengths, spec, 'right')
                if startGapLength > maxGapLength or endGapLength > maxGapLength:
                    gapMask.append(False)
                else:
                    gapMask.append(True)
        gapMask = np.array(gapMask)
        self.applyMask(gapMask, "Removed spectra with gaps longer than %d wavelength bins,\
                                 and spectra with gaps at the beginning or end."%(maxGapLength))

        wvlMask = np.logical_and(self.wavelengths > self.minwvl, self.wavelengths < self.maxwvl)
        filtWavelengths = self.wavelengths[wvlMask]
        filtSpectraMatrix = np.ndarray((self.spectraMatrix.shape[0], len(filtWavelengths)))


        return

    def interpSpec(self):

        wvlMask = np.logical_and(self.wavelengths > self.minwvl, self.wavelengths < self.maxwvl)
        filtWavelengths = self.wavelengths[wvlMask]
        #print filtWavelengths
        filtSpectraMatrix = np.ndarray((self.spectraMatrix.shape[0], len(filtWavelengths)))

        for i,spec in enumerate(self.spectraMatrix):
            startWvl = self.wavelengths[np.where(self.wavelengths > self.minwvl)[0][0]]
            endWvl = self.wavelengths[np.where(self.wavelengths < self.maxwvl)[0][-1]]
            #print startWvl, endWvl

            startGapLength, startfiniteWvl = findGapLength(startWvl, self.wavelengths, spec, 'left')
            endGapLength, endfiniteWvl = findGapLength(endWvl, self.wavelengths, spec, 'right')
            print startfiniteWvl, endfiniteWvl

            wvlMask = np.logical_and(self.wavelengths >= startfiniteWvl, self.wavelengths <= endfiniteWvl)
            finiteMask = np.isfinite(spec)
            mask = np.logical_and(wvlMask, finiteMask)
            interp_fn = interp1d(self.wavelengths[mask], spec[mask])
            #print filtWavelengths[0], filtWavelengths[-1]
            #print spec[0], spec[-1]
            filtSpectraMatrix[i] = interp_fn(filtWavelengths)
        self.wavelengths = filtWavelengths
        self.spectraMatrix = filtSpectraMatrix
        return





# This Method smooths the broadline Ic spectra and outputs
# a plot of the smoothed vs unsmoothed spectra for verification.

    def smoothIcBL(self, smoothMask, vel_cut):
        """
This method smoothes the IcBL spectra because they are noisier than the other types. It uses the IDL code SNspecFFTsmooth.pro from the nyusngroup public git repository: https://github.com/nyusngroup/SESNspectraLib. Smoothing is done using cut_vel = 3000 km/s as recommended by the authors for IcBL. This method replaces the noisy IcBL spectra loaded from SNID and stored in self.spectraMatrix with the smoothed spectra outputted by the IDL code.
        """
        #IcBLMask = np.array([np.array_equal(arr, np.array([3,4])) for arr in self.sneTypes])
        idl = pidly.IDL()
        idl('.COMPILE type default lmfit linear_fit powerlaw_fit integ binspec SNspecFFTsmooth')
        IcBLSmoothedMatrix = []
        IcBLPreSmooth = []
        IcBLPreSmoothIDL = []
        sepvels = []
        sepvels_no_pad = []
        
        for i in range(len(smoothMask[smoothMask == True])):
            specName = self.sneNames[smoothMask][i]
            print specName
            specPath = self.snidDirPath + specName +'.lnw'
            skiprow = self.skiprows[smoothMask][i]
            usecol = self.phaseCols[smoothMask][i] + 1 # add 1 because SNID file has a wavelength column
            s = np.loadtxt(specPath, skiprows=skiprow, usecols=(0,usecol))
            if i==0:
                IcBLWvl = s[:,0]
            with open('tmp_spec.txt', 'w') as f:
                for j in range(s.shape[0]):
                    f.write('        %.4f        %.7f\n'%(s[j,0],s[j,1]+0.0))
                f.close()

            #spec = self.spectraMatrix[smoothMask][i]
            #with open('tmp_spec_wvlcut.txt', 'w') as f:
            #    for j in range(len(self.wavelengths)):
            #        f.write('        %.4f        %.7f\n'%(self.wavelengths[j], spec[j]))
            #    f.close()


            idl('readcol, "tmp_spec.txt", w, f')
            idlCmd = 'SNspecFFTsmooth, w, f, '+str(vel_cut[i])+', f_ft, f_std, sep_vel'
            idl(idlCmd)
           #idl('SNspecFFTsmooth, w, f, 3000, f_ft, f_std, sep_vel')
            IcBLPreSmooth.append(s[:,1])
            IcBLSmoothedMatrix.append(idl.f_ft)
            IcBLPreSmoothIDL.append(idl.f)
            sepvels.append(idl.sep_vel)
            
            #idl('readcol, "tmp_spec_wvlcut.txt", w_pad, f_pad')
            #idlCmd = 'SNspecFFTsmooth, w_pad, f_pad, '+str(vel_cut[i])+', f_ft_pad, f_std_pad, sep_vel_pad'
            #idl(idlCmd)
            #sepvels_no_pad.append(idl.sep_vel_pad)




        IcBLPreSmooth = np.array(IcBLPreSmooth)
        IcBLSmoothedMatrix = np.array(IcBLSmoothedMatrix)
        IcBLPreSmoothIDL = np.array(IcBLPreSmoothIDL)

        smoothMean = np.nanmean(IcBLSmoothedMatrix, axis=1)
        smoothStd = np.nanstd(IcBLSmoothedMatrix, axis=1)

        originalSmoothSpectra = np.copy(IcBLSmoothedMatrix)
        IcBLSmoothedMatrixT = (IcBLSmoothedMatrix.T - smoothMean) / smoothStd
        IcBLSmoothedMatrix = IcBLSmoothedMatrixT.T

        preSmoothMean = np.nanmean(IcBLPreSmooth, axis=1)
        preSmoothStd = np.nanstd(IcBLPreSmooth, axis=1)
        originalPreSmoothSpectra = np.copy(IcBLPreSmooth)
        IcBLPreSmoothT = (IcBLPreSmooth.T - preSmoothMean) / preSmoothStd
        IcBLPreSmooth = IcBLPreSmoothT.T
        
        smoothspecvec = []

        f = plt.figure(figsize=(15,50))
        nshow = IcBLSmoothedMatrix.shape[0]
        for i in range(nshow):
            plt.subplot(nshow, 1, i + 1)
            name = self.sneNames[smoothMask][i]
            plt.plot(IcBLWvl, IcBLSmoothedMatrix[i], label='smoothed '+name, color=self.IcBL_color)
            plt.plot(IcBLWvl, IcBLPreSmooth[i], label='pre smoothed '+name, color=self.Ic_color)
            if i == 0:
                plt.title('Smoothed IcBL spectra %d$\pm$%d'%(self.loadPhase, self.phaseWidth))
            if i == nshow - 1:
                plt.xlabel('Wavelength (Angstroms)')
            plt.ylabel('Rel Flux')
            plt.legend(fontsize=12)
        for i, sn in enumerate(self.sneNames[smoothMask]):
            print sn
            smoothSpec = IcBLSmoothedMatrix[i]
            mask = np.logical_and(IcBLWvl > self.minwvl, IcBLWvl < self.maxwvl)
            smoothSpec = smoothSpec[mask]
            ind = np.where(self.sneNames == sn)[0][0]
            #self.spectraMatrix[ind] = smoothSpec
            smoothspecvec.append(smoothSpec)


        return f, np.array(sepvels), np.array(smoothspecvec), np.array(sepvels_no_pad)

# Preprocessing replaces 0.0 values with NaN. It also removes the mean of each spectrum
# and scales each spectrum to have unitary std.
    def preprocess(self):
        """
Preprocessing replaces flux values of 0.0 with NaN. It also removes the mean of each spectrum and scales each spectrum to have standard deviation = 1.
        """
        for i in range(self.spectraMatrix.shape[0]):
            self.spectraMatrix[i][self.spectraMatrix[i] == 0] = np.nan
        spectraMean = np.nanmean(self.spectraMatrix, axis=1)
        spectraStd = np.nanstd(self.spectraMatrix, axis=1)
        spectraMatrixT = (self.spectraMatrix.T - spectraMean)/spectraStd
        self.spectraMatrix = spectraMatrixT.T
        self.spectraMean = spectraMean
        self.spectraStd = spectraStd
        return



    def wavelengthRebin(self, smoothing):
        """
Rebins wavelength to have lower resolution. Wavelength is in logspace.
Arguments:
    smoothing -- Number of wavelength bins to combine by averaging
Example: If there are 100 wavelength bins in logspace, then setting smoothing = 2 will split the original 100 bins into 50 new bins of length 2 original bins each, then calculate a flux for the new larger bin by averaging the original flux bins. 
        """
        nrows, ncols = self.spectraMatrix.shape
        tmp = np.reshape(self.spectraMatrix, (nrows, ncols/smoothing, smoothing))
        self.spectraMatrix = np.nanmean(tmp, axis=2)
        wvrows = self.wavelengths.shape[0]
        wvcols = 1
        wvtemp = np.reshape(self.wavelengths, (wvrows/smoothing, 1, smoothing))
        self.wavelengths = np.nanmean(wvtemp, axis=2)
        return

# This method takes a user specified mask, and applies it to all the maskable 
# attributes of a NovaPCA instance. If the user sets savecopy=True, then this 
# method first copies the original NovaPCA instance before applying the mask
# and returns the old instance to the user.
    def applyMask(self, mask, doc, savecopy=False):
        """
This method takes a user specified mask, and applies it to all the maskable attributes of a NovaPCA instance. If the user sets savecopy=True, then this method first makes a deep copy of the original NovaPCA instance before applying the mask, and returns the preMasked instance to the user.
Arguments:
    mask -- Numpy boolean array.
    doc -- Documentation string for the mask that is being applied.
    savecopy -- Boolean, whether to save the preMasked NovaPCA object.
        """
        self.maskDocs.append(doc)
        if savecopy:
            preMask = copy.deepcopy(self)
        for attr in self.maskAttributes:
            attrObj = getattr(self, attr)
            if not attrObj is None:
                setattr(self, attr, attrObj[mask])
        if savecopy: 
            return preMask
        return

# The save method pickles the NovaPCA object.
    def save(self, filename):
        """
The save method pickles the NovaPCA object.
        """
        f = open(filename, 'wb')
        pickle.dump(self, f)
        f.close()
        return

# The load method loads a saved pickle file.
    def load(self, filename):
        """
Loads a pickled NovaPCA object. Prints all of the applied mask doc strings so that the user is informed as to which masks have already been applied to the object.
        """
        f = open(filename, 'rb')
        loadSelf = pickle.load(f)
        f.close()
        for doc in loadSelf.maskDocs:
            print "Applied Mask: "
            print doc
            print ''
        return loadSelf

# Calculate PCA decomposition
    def calculatePCA(self):
        """
Calculates the PCA decomposition. PCA coefficients are stored in NovaPCA.pcaCoeffMatrix, but the sklearn PCA() object is also stored in case the user wants to experiment. This is accessible in NovaPCA.sklearnPCA attribute.
        """
        pca = PCA()
        pca.fit(self.spectraMatrix)
        self.sklearnPCA = pca
        self.evecs = pca.components_
        self.evals = pca.explained_variance_ratio_
        self.evals_cs = self.evals.cumsum()
        self.pcaCoeffMatrix = np.dot(self.evecs, self.spectraMatrix.T).T
        return

# Plot TSNE embedding

    def plotTSNE(self, nPCAComponents):
        """
Calculates the TSNE embedding of a PCA decomposition in a 2 dimensional space.
        """
        f = plt.figure()
        model = TSNE(n_components=2, random_state=0)
        tsneSpec = model.fit_transform(self.pcaCoeffMatrix[:,0:nPCAComponents])

        IIbMask, IbMask, IcMask, IcBLMask = getSNeTypeMasks(self.sneTypes)
        plt.scatter(tsneSpec[:,0][IIbMask], tsneSpec[:,1][IIbMask], color=self.IIb_color)
        plt.scatter(tsneSpec[:,0][IbMask], tsneSpec[:,1][IbMask], color=self.Ib_color)
        plt.scatter(tsneSpec[:,0][IcMask], tsneSpec[:,1][IcMask], color=self.Ic_color)
        plt.scatter(tsneSpec[:,0][IcBLMask], tsneSpec[:,1][IcBLMask], color=self.IcBL_color)
        plt.title('TSNE Projection from PCA')
        plt.xlabel('TSNE Component 0')
        plt.ylabel('TSNE Component 1')
        return f



    def pcaPlot(self, pcax, pcay, figsize, purity=False, std_rad=None):
        f = plt.figure(figsize=figsize)
        ax = plt.gca()
        red_patch = mpatches.Patch(color=self.Ic_color, label='Ic')
        cyan_patch = mpatches.Patch(color=self.Ib_color, label='Ib')
        black_patch = mpatches.Patch(color=self.IcBL_color, label='IcBL')
        green_patch = mpatches.Patch(color=self.IIb_color, label='IIb')

        IIbMask, IbMask, IcMask, IcBLMask = getSNeTypeMasks(self.sneTypes)

        x = self.pcaCoeffMatrix[:,pcax-1]
        y = self.pcaCoeffMatrix[:,pcay-1]

        #centroids
        IIbxmean = np.mean(x[IIbMask])
        IIbymean = np.mean(y[IIbMask])
        Ibxmean = np.mean(x[IbMask])
        Ibymean = np.mean(y[IbMask])
        Icxmean = np.mean(x[IcMask])
        Icymean = np.mean(y[IcMask])
        IcBLxmean = np.mean(x[IcBLMask])
        IcBLymean = np.mean(y[IcBLMask])
        plt.scatter(IIbxmean, IIbymean, color=self.IIb_color, alpha=0.5, s=100, marker='x')
        plt.scatter(Ibxmean, Ibymean, color=self.Ib_color, alpha=0.5, s=100, marker='x')
        plt.scatter(Icxmean, Icymean, color=self.Ic_color, alpha=0.5, s=100, marker='x')
        plt.scatter(IcBLxmean, IcBLymean, color=self.IcBL_color, alpha=0.5, s=100, marker='x')

        if purity:
            ncomp_arr = [pcax, pcay]
            keys, purity_rad_arr = self.purityEllipse(std_rad, ncomp_arr)
            IIbrad = purity_rad_arr[0]
            Ibrad = purity_rad_arr[1]
            IcBLrad = purity_rad_arr[2]
            Icrad = purity_rad_arr[3]

            ellipse_IIb = mpatches.Ellipse((IIbxmean, IIbymean),2*IIbrad[0],2*IIbrad[1], color=self.IIb_color, alpha=0.1)
            ellipse_Ib = mpatches.Ellipse((Ibxmean, Ibymean),2*Ibrad[0],2*Ibrad[1], color=self.Ib_color, alpha=0.1)
            ellipse_Ic = mpatches.Ellipse((Icxmean, Icymean),2*Icrad[0],2*Icrad[1], color=self.Ic_color, alpha=0.1)
            ellipse_IcBL = mpatches.Ellipse((IcBLxmean, IcBLymean),2*IcBLrad[0],2*IcBLrad[1], color=self.IcBL_color, alpha=0.1)

            ax.add_patch(ellipse_IIb)
            ax.add_patch(ellipse_Ib)
            ax.add_patch(ellipse_Ic)
            ax.add_patch(ellipse_IcBL)

        plt.scatter(x[IIbMask], y[IIbMask], color=self.IIb_color, alpha=1)
        plt.scatter(x[IbMask], y[IbMask], color=self.Ib_color, alpha=1)
        plt.scatter(x[IcMask], y[IcMask], color=self.Ic_color, alpha=1)
        plt.scatter(x[IcBLMask], y[IcBLMask], color=self.IcBL_color, alpha=1)
        #for i, name in enumerate(self.sneNames[IcBLMask]):
        #    plt.text(x[IcBLMask][i], y[IcBLMask][i], name)

        plt.xlim((np.min(x)-2,np.max(x)+2))
        plt.ylim((np.min(y)-2,np.max(y)+2))

        plt.ylabel('PCA Comp %d'%(pcay),fontsize=20)
        plt.xlabel('PCA Comp %d'%(pcax), fontsize=20)
#        plt.axis('off')
        plt.legend(handles=[red_patch, cyan_patch, black_patch, green_patch], fontsize=18)
        #plt.title('PCA Space Separability of IcBL and IIb SNe (Phase %d$\pm$%d Days)'%(self.loadPhase, self.phaseWidth),fontsize=22)
        plt.minorticks_on()
        plt.tick_params(
                    axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    labelsize=20) # labels along the bottom edge are off

        return f


# Plot 2D Corner plot of PCA components


    def purityEllipse(self, std_rad, ncomp_array):
        ncomp_array = np.array(ncomp_array) - 1
        IIbMask, IbMask, IcMask, IcBLMask = getSNeTypeMasks(self.sneTypes)
        maskDict = {'IIb':IIbMask, 'Ib':IbMask, 'IcBL':IcBLMask, 'Ic':IcMask}
        keys = ['IIb', 'Ib', 'IcBL', 'Ic']
        masks = [IIbMask, IbMask, IcBLMask, IcMask]
        purity_rad_arr = []
        for key,msk in zip(keys,masks):
            centroid = np.mean(self.pcaCoeffMatrix[:,ncomp_array][msk], axis=0)
            print 'centroid', centroid
            dist_from_centroid = np.abs(self.pcaCoeffMatrix[:,ncomp_array][msk] - centroid)
            mean_dist_from_centroid = np.mean(dist_from_centroid, axis=0)
            print 'mean dist from centroid: ', mean_dist_from_centroid
            std_dist_all_components = np.std(dist_from_centroid, axis=0)
            print 'std dist from centroid: ', std_dist_all_components
            purity_rad_all = mean_dist_from_centroid + std_rad * std_dist_all_components
            print 'purity rad all components: ', purity_rad_all
            purity_rad_arr.append(purity_rad_all)
            

            ellipse_cond = np.sum(np.power((self.pcaCoeffMatrix[:,ncomp_array] - centroid), 2)/\
                                  np.power(purity_rad_all, 2), axis=1)
            print 'ellipse condition: ', ellipse_cond
            purity_msk = ellipse_cond < 1

            print key
            print 'purity radius: ', purity_rad_all
            print '# of SNe within purity ellipse for type '+key+': ',np.sum(purity_msk)
            names_within_purity_rad = self.sneNames[purity_msk]
            correct_names = self.sneNames[msk]
            correct_msk = np.isin(names_within_purity_rad, correct_names)
            print '# of correct SNe '+key+': ', np.sum(correct_msk)
        return keys, purity_rad_arr


    def purity(self, std_rad, ncomp_array):
        IIbMask, IbMask, IcMask, IcBLMask = getSNeTypeMasks(self.sneTypes)
        maskDict = {'IIb':IIbMask, 'Ib':IbMask, 'IcBL':IcBLMask, 'Ic':IcMask}
        keys = ['IIb', 'Ib', 'IcBL', 'Ic']
        masks = [IIbMask, IbMask, IcBLMask, IcMask]
        purity_rad_arr = []
        for key,msk in zip(keys,masks):
            mean = np.mean(self.pcaCoeffMatrix[:,ncomp_array][msk], axis=0)
            d = distance.cdist(self.pcaCoeffMatrix[:,ncomp_array][msk], np.array([mean]), 'euclidean')
            dstd = np.std(d)
            dmean = np.mean(d)

            purity_rad = dmean + std_rad * dstd
            purity_rad_arr.append(purity_rad)
            d_all = distance.cdist(self.pcaCoeffMatrix[:,ncomp_array], np.array([mean]), 'euclidean')
            purity_msk = d_all < purity_rad
            purity_msk = purity_msk.flatten()
            print key
            print 'purity radius: ', purity_rad
            print '# of SNe within purity radius for type '+key+': ',np.sum(purity_msk)
            names_within_purity_rad = self.sneNames[purity_msk]
            correct_names = self.sneNames[msk]
            correct_msk = np.isin(names_within_purity_rad, correct_names)
            print '# of correct SNe '+key+': ', np.sum(correct_msk)
        return keys, purity_rad_arr





    def cornerplotPCA(self, ncomp, figsize):
        """
Plots the 2D marginalizations of the PCA decomposition in a corner plot.
Arguments:
    ncomp -- Number of PCA components to include in the 2D marginalization. It is best to ignore the high order components that only capture noise.
    figsize -- Size of the figure.
        """
        red_patch = mpatches.Patch(color=self.Ic_color, label='Ic')
        cyan_patch = mpatches.Patch(color=self.Ib_color, label='Ib')
        black_patch = mpatches.Patch(color=self.IcBL_color, label='IcBL Smoothed')
        green_patch = mpatches.Patch(color=self.IIb_color, label='IIb')

        IIbMask, IbMask, IcMask, IcBLMask = getSNeTypeMasks(self.sneTypes)

        f = plt.figure(figsize=figsize)
        for i in range(ncomp):
            for j in range(ncomp):
                if i > j:
                    plotNumber = ncomp * i + j + 1
                    plt.subplot(ncomp, ncomp, plotNumber)
                    x = self.pcaCoeffMatrix[:,i]
                    y = self.pcaCoeffMatrix[:,j]

                    #centroids
                    IIbxmean = np.mean(x[IIbMask])
                    IIbymean = np.mean(y[IIbMask])
                    Ibxmean = np.mean(x[IbMask])
                    Ibymean = np.mean(y[IbMask])
                    Icxmean = np.mean(x[IcMask])
                    Icymean = np.mean(y[IcMask])
                    IcBLxmean = np.mean(x[IcBLMask])
                    IcBLymean = np.mean(y[IcBLMask])
                    plt.scatter(IIbymean, IIbxmean, color=self.IIb_color, alpha=0.5, s=100)
                    plt.scatter(Ibymean, Ibxmean, color=self.Ib_color, alpha=0.5, s=100)
                    plt.scatter(Icymean, Icxmean, color=self.Ic_color, alpha=0.5, s=100)
                    plt.scatter(IcBLymean, IcBLxmean, color=self.IcBL_color, alpha=0.5, s=100)

                    plt.scatter(y[IIbMask], x[IIbMask], color=self.IIb_color, alpha=1)
                    plt.scatter(y[IbMask], x[IbMask], color=self.Ib_color, alpha=1)
                    plt.scatter(y[IcMask], x[IcMask], color=self.Ic_color, alpha=1)
                    plt.scatter(y[IcBLMask], x[IcBLMask], color=self.IcBL_color, alpha=1)

                    plt.xlim((np.min(self.pcaCoeffMatrix[:,j])-2,np.max(self.pcaCoeffMatrix[:,j])+2))
                    plt.ylim((np.min(self.pcaCoeffMatrix[:,i])-2,np.max(self.pcaCoeffMatrix[:,i])+2))

                    if j == 0:
                        plt.ylabel('PCA Comp %d'%(i+1))
                    if i == ncomp - 1:
                        plt.xlabel('PCA Comp %d'%(j+1))
        plt.subplot(5,5,9)#########################################################
        plt.axis('off')
        plt.legend(handles=[red_patch, cyan_patch, black_patch, green_patch])
        plt.text(-3.0,1.3,'Smoothed IcBL PCA Component 2D Marginalizations (Phase %d$\pm$%d Days)'%(self.loadPhase, self.phaseWidth),fontsize=16)
        return f



    def plotEigenspectraGrid(self, figsize, nshow, ylim=None, fontsize=16):
        f = plt.figure(figsize=figsize)
        hostgrid = gridspec.GridSpec(3,1)
        hostgrid.update(hspace=0.2)

        eiggrid = gridspec.GridSpecFromSubplotSpec(nshow, 1, subplot_spec=hostgrid[:2,0], hspace=0)

        for i, ev in enumerate(self.evecs[:nshow]):
            ax = plt.subplot(eiggrid[i,0])
            ax.plot(self.wavelengths, ev, color=self.IcBL_color)

            trans2 = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
            ax.text(0.02,0.85, "(PCA%d, %.0f"%(i+1, 100*self.evals_cs[i])+'%)', horizontalalignment='left',\
                    verticalalignment='center', fontsize=fontsize, transform=trans2)
            ax.tick_params(axis='both',which='both',labelsize=fontsize)
            if not ylim is None:
                ax.set_ylim(ylim)
            if i > -1:
                yticks = ax.yaxis.get_major_ticks()
                yticks[-1].set_visible(False)

            if i == 0:
                # Balmer lines
                trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                trans2 = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)

                ax.text(0.02,1.05, "(PCA#, Cum. Var.)", fontsize=fontsize, horizontalalignment='left',\
                        verticalalignment='center', transform=trans2)

                ax.axvspan(6213, 6366, alpha=0.1, color=self.H_color) #H alpha -9000 km/s to -16000 km/s
                s = r'$\alpha$'
                xcord = (6213+6366)/2.0
                ax.text(xcord, 1.05, 'H'+s, fontsize=fontsize, horizontalalignment='center',\
                        verticalalignment='center',transform=trans)
                ax.axvspan(4602, 4715, alpha=0.1, color=self.H_color) #H Beta -9000 km/s to-16000 km/s
                s = r'$\beta$'
                xcord = (4602+4715)/2.0
                ax.text(xcord, 1.05, 'H'+s, fontsize=fontsize, horizontalalignment='center',\
                        verticalalignment='center',transform=trans)


                ax.axvspan(5621, 5758, alpha=0.1, color=self.He_color) #HeI5876 -6000 km/s to -13000 km/s
                ax.text((5621+5758)/2.0, 1.05, 'HeI5876', fontsize=fontsize, horizontalalignment='center',\
                        verticalalignment='center', transform=trans)
                ax.axvspan(6388, 6544, alpha=0.1, color=self.He_color)
                ax.text((6388+6544)/2.0, 1.05, 'HeI6678', fontsize=fontsize, horizontalalignment='center',\
                        verticalalignment='center', transform=trans)
                ax.axvspan(6729, 6924, alpha=0.1, color=self.He_color)
                ax.text((6729+6924)/2.0, 1.05, 'HeI7065', fontsize=fontsize, horizontalalignment='center',\
                        verticalalignment='center', transform=trans)
            if i > 0:
                # Balmer lines
                trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                ax.axvspan(6213, 6366, alpha=0.1, color=self.H_color) #H alpha -9000 km/s to -16000 km/s
                ax.axvspan(4602, 4715, alpha=0.1, color=self.H_color) #H Beta -9000 km/s to-16000 km/s
                ax.axvspan(5621, 5758, alpha=0.1, color=self.He_color) #HeI5876 -6000 km/s to -13000 km/s
                ax.axvspan(6388, 6544, alpha=0.1, color=self.He_color)
                ax.axvspan(6729, 6924, alpha=0.1, color=self.He_color)


            if i == nshow - 1:
                ax.set_xlabel("Wavelength", fontsize=fontsize)

        ax = plt.subplot(hostgrid[-1])
        ax.boxplot(self.pcaCoeffMatrix)
        ax.set_xlabel('PCA Component #', fontsize=fontsize)
        ax.set_ylabel('PCA Coefficient Value', fontsize=fontsize)
        ax.tick_params(axis='both', which='both', labelsize=fontsize)
        ax.axhline(y=0, color=self.Ic_color)
        xticklabels = ax.xaxis.get_majorticklabels()
        xticklabels[0].set_visible
        for i, tick in enumerate(xticklabels):
            if i%4 != 0:
                tick.set_visible(False)
            
        #f.text(0.07, 2.0/3.0, 'Relative Flux', verticalalignment='center', rotation='vertical', fontsize=16)
        return f, hostgrid

        

    def plotEigenspectraFinal(self, figsize, nshow, ylim=None, fontsize=16):
        """
Plots the eigenspectra calculated by PCA.
Arguments:
    figsize -- Size of the figure.
    nshow -- Number of eigenspectra to show in figure. The high order PCA components are noise.
        """
        f = plt.figure(figsize=figsize)
        plt.tick_params(axis='both', which='both', bottom='off', top='off',\
                            labelbottom='off', right='off', left='off', labelleft='off')
        f.subplots_adjust(hspace=0, top=0.95, bottom=0.1, left=0.12, right=0.93)
        for i, ev in enumerate(self.evecs[:nshow]):
            ax = f.add_subplot(nshow, 1, 1+i)
            #ax.plot(self.wavelengths, ev, label="PCA%d, %.0f"%(i, 100*self.evals_cs[i])+'%', color=self.IcBL_color)
            ax.plot(self.wavelengths, ev, color=self.IcBL_color)

            trans2 = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
            ax.text(0.02,0.85, "(PCA%d, %.0f"%(i+1, 100*self.evals_cs[i])+'%)', horizontalalignment='left',\
                    verticalalignment='center', fontsize=fontsize, transform=trans2)
            ax.tick_params(axis='both',which='both',labelsize=fontsize)
            if not ylim is None:
                ax.set_ylim(ylim)
            if i > -1:
                yticks = ax.yaxis.get_major_ticks()
                yticks[-1].set_visible(False)

            if i == 0:
                # Balmer lines
                trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                trans2 = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)

                ax.text(0.02,1.05, "(PCA#, Cum. Var.)", fontsize=fontsize, horizontalalignment='left',\
                        verticalalignment='center', transform=trans2)

                ax.axvspan(6213, 6366, alpha=0.1, color=self.H_color) #H alpha -9000 km/s to -16000 km/s
                s = r'$\alpha$'
                xcord = (6213+6366)/2.0
                ax.text(xcord, 1.05, 'H'+s, fontsize=fontsize, horizontalalignment='center',\
                        verticalalignment='center',transform=trans)
                ax.axvspan(4602, 4715, alpha=0.1, color=self.H_color) #H Beta -9000 km/s to-16000 km/s
                s = r'$\beta$'
                xcord = (4602+4715)/2.0
                ax.text(xcord, 1.05, 'H'+s, fontsize=fontsize, horizontalalignment='center',\
                        verticalalignment='center',transform=trans)


                ax.axvspan(5621, 5758, alpha=0.1, color=self.He_color) #HeI5876 -6000 km/s to -13000 km/s
                ax.text((5621+5758)/2.0, 1.05, 'HeI5876', fontsize=fontsize, horizontalalignment='center',\
                        verticalalignment='center', transform=trans)
                ax.axvspan(6388, 6544, alpha=0.1, color=self.He_color)
                ax.text((6388+6544)/2.0, 1.05, 'HeI6678', fontsize=fontsize, horizontalalignment='center',\
                        verticalalignment='center', transform=trans)
                ax.axvspan(6729, 6924, alpha=0.1, color=self.He_color)
                ax.text((6729+6924)/2.0, 1.05, 'HeI7065', fontsize=fontsize, horizontalalignment='center',\
                        verticalalignment='center', transform=trans)
            if i > 0:
                # Balmer lines
                trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                ax.axvspan(6213, 6366, alpha=0.1, color=self.H_color) #H alpha -9000 km/s to -16000 km/s
                ax.axvspan(4602, 4715, alpha=0.1, color=self.H_color) #H Beta -9000 km/s to-16000 km/s
                ax.axvspan(5621, 5758, alpha=0.1, color=self.He_color) #HeI5876 -6000 km/s to -13000 km/s
                ax.axvspan(6388, 6544, alpha=0.1, color=self.He_color)
                ax.axvspan(6729, 6924, alpha=0.1, color=self.He_color)


                
            #if i == 0:
                #plt.title('PCA Eigenspectra Phase %d$\pm$%d'%(self.loadPhase, self.phaseWidth), fontsize=18)
            if i == nshow - 1:
                plt.xlabel("Wavelength", fontsize=fontsize)
            #plt.ylabel("Rel Flux", fontsize=16)

            #plt.legend(loc='lower left', fontsize=16)
        return f



# Plot reconstructed spectra

    def reconstructSpectra(self, nrecon, nPCAComponents, snname=None, fontsize=16):
        """
Reconstructs spectra using the PCA decomposition.
Arguments:
    nrecon -- Number of randomly chosen spectra to reconstruct.
    nPCAComponents -- Iterable list of numbers of PCA components to try using for the reconstruction.
        """
        randomSpec = np.random.randint(0,self.spectraMatrix.shape[0], nrecon)
        if not snname is None:
            randomSpec = np.where(self.sneNames == snname)[0]

        self.sampleMean = np.nanmean(self.spectraMatrix, axis=0)
        plt.clf()
        for j, spec in enumerate(randomSpec):
            specName = self.sneNames[spec]
            trueSpec = self.spectraMatrix[spec]
            pcaCoeff = np.dot(self.evecs, (trueSpec - self.sampleMean))
            f = plt.figure(j, figsize=(15,20))
            plt.tick_params(axis='both', which='both', bottom='off', top='off',\
                            labelbottom='off', labelsize=40, right='off', left='off', labelleft='off')
            #plt.title(specName +' PCA Reconstruction',fontsize=16)
            f.subplots_adjust(hspace=0, top=0.95, bottom=0.1, left=0.12, right=0.93)

            for i, n in enumerate(nPCAComponents):
                ax = f.add_subplot(411 + i)
                ax.plot(self.wavelengths, trueSpec, '-', c='gray')
                ax.plot(self.wavelengths, self.sampleMean + (np.dot(pcaCoeff[:n], self.evecs[:n])), '-k')
                ax.tick_params(axis='both',which='both',labelsize=20)
                if i < len(nPCAComponents) - 1:
                    plt.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    labelbottom='off') # labels along the bottom edge are off
                ax.set_ylim(-5,5)
                #ax.set_ylabel('flux', fontsize=30)

                if i == 0:
                    # Balmer lines
                    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                    trans2 = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)

                    ax.text(0.02,1.05, "(N PCA, % Var.)", fontsize=fontsize, horizontalalignment='left',\
                            verticalalignment='center', transform=trans2)

                    ax.axvspan(6213, 6366, alpha=0.1, color=self.H_color) #H alpha -9000 km/s to -16000 km/s
                    s = r'$\alpha$'
                    xcord = (6213+6366)/2.0
                    ax.text(xcord, 1.05, 'H'+s, fontsize=fontsize, horizontalalignment='center',\
                            verticalalignment='center',transform=trans)
                    ax.axvspan(4602, 4715, alpha=0.1, color=self.H_color) #H Beta -9000 km/s to-16000 km/s
                    s = r'$\beta$'
                    xcord = (4602+4715)/2.0
                    ax.text(xcord, 1.05, 'H'+s, fontsize=fontsize, horizontalalignment='center',\
                            verticalalignment='center',transform=trans)


                    ax.axvspan(5621, 5758, alpha=0.1, color=self.He_color) #HeI5876 -6000 km/s to -13000 km/s
                    ax.text((5621+5758)/2.0, 1.05, 'HeI5876', fontsize=fontsize, horizontalalignment='center',\
                            verticalalignment='center', transform=trans)
                    ax.axvspan(6388, 6544, alpha=0.1, color=self.He_color)
                    ax.text((6388+6544)/2.0, 1.05, 'HeI6678', fontsize=fontsize, horizontalalignment='center',\
                            verticalalignment='center', transform=trans)
                    ax.axvspan(6729, 6924, alpha=0.1, color=self.He_color)
                    ax.text((6729+6924)/2.0, 1.05, 'HeI7065', fontsize=fontsize, horizontalalignment='center',\
                            verticalalignment='center', transform=trans)
                if i > 0:
                    # Balmer lines
                    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                    ax.axvspan(6213, 6366, alpha=0.1, color=self.H_color) #H alpha -9000 km/s to -16000 km/s
                    ax.axvspan(4602, 4715, alpha=0.1, color=self.H_color) #H Beta -9000 km/s to-16000 km/s
                    ax.axvspan(5621, 5758, alpha=0.1, color=self.He_color) #HeI5876 -6000 km/s to -13000 km/s
                    ax.axvspan(6388, 6544, alpha=0.1, color=self.He_color)
                    ax.axvspan(6729, 6924, alpha=0.1, color=self.He_color)
                if n == 0:
                    text = 'mean'
                elif n == 1:
                    text = "1 component\n"
                    text += r"$(\sigma^2_{tot} = %.2f)$" % self.evals_cs[n - 1]
                    
                else:
                    text = "%i components\n" % n
                    text += r"$(\sigma^2_{tot} = %.2f)$" % self.evals_cs[n - 1]
                    text = '(%i, %.0f'%(n, 100*self.evals_cs[n-1])+'%)'
                ax.text(0.02, 0.93, text, fontsize=20,ha='left', va='top', transform=ax.transAxes)
                f.axes[-1].set_xlabel(r'${\rm wavelength\ (\AA)}$',fontsize=30)
        return f







# Plot eigenspectra

    def plotEigenspectra(self, figsize, nshow):
        """
Plots the eigenspectra calculated by PCA.
Arguments:
    figsize -- Size of the figure.
    nshow -- Number of eigenspectra to show in figure. The high order PCA components are noise.
        """
        f = plt.figure(figsize=figsize)
        for i, ev in enumerate(self.evecs[:nshow]):
            plt.subplot(nshow, 1, i + 1)
            #plt.plot(self.wavelengths, self.evals[i] * ev, label="component: %d, %.2f"%(i, self.evals_cs[i]))
            plt.plot(self.wavelengths, ev, label="component: %d, %.2f"%(i, self.evals_cs[i]))
            if i == 0:
                plt.title('PCA Eigenspectra Phase %d$\pm$%d'%(self.loadPhase, self.phaseWidth), fontsize=18)
            if i == nshow - 1:
                plt.xlabel("Wavelength", fontsize=16)
            plt.ylabel("Rel Flux", fontsize=16)
            plt.legend(fontsize=12)
        return f

# Plot spectra

    def plotSpectra(self, figsize, alpha):
        """
Plots the spectra stored in NovaPCA.spectraMatrix attribute.
Arguments:
    figsize -- Size of figure.
    alpha -- Matplotlib alpha parameter.
        """
        f = plt.figure(figsize=figsize)
        for i, spec in enumerate(self.spectraMatrix):
            if not i % 10:
                plt.plot(self.wavelengths, spec + i*2, alpha=1.0)
            else:
                plt.plot(self.wavelengths, spec + i*2, alpha=alpha)
        plt.xlabel("Wavelengths (Angstroms)")
        plt.title("All Spectra")
        return f





#* SN Ia
#      typename(1,1) = 'Ia'      ! first element is name of type
#      typename(1,2) = 'Ia-norm' ! subtypes follow...(normal, peculiar, etc.)
#      typename(1,3) = 'Ia-91T'
#      typename(1,4) = 'Ia-91bg'
#      typename(1,5) = 'Ia-csm'
#      typename(1,6) = 'Ia-pec'
#      typename(1,7) = 'Ia-99aa'
#      typename(1,8) = 'Ia-02cx'

#* SN Ib      
#      typename(2,1) = 'Ib'
#      typename(2,2) = 'Ib-norm'
#      typename(2,3) = 'Ib-pec'
#      typename(2,4) = 'IIb'     ! IIb is not included in SNII
#      typename(2,5) = 'Ib-n'    ! Ib-n can be regarded as a kind of Ib-pec 

#* SN Ic
#      typename(3,1) = 'Ic'
#      typename(3,2) = 'Ic-norm'
#      typename(3,3) = 'Ic-pec'
#      typename(3,4) = 'Ic-broad'

#* SN II
#      typename(4,1) = 'II'
#      typename(4,2) = 'IIP'     ! IIP is the "normal" SN II
#      typename(4,3) = 'II-pec'
#      typename(4,4) = 'IIn'
#      typename(4,5) = 'IIL'

#* NotSN
#      typename(5,1) = 'NotSN'
#      typename(5,2) = 'AGN'
#      typename(5,3) = 'Gal'
#      typename(5,4) = 'LBV'
#      typename(5,5) = 'M-star'
#      typename(5,6) = 'QSO'
#      typename(5,7) = 'C-star'
