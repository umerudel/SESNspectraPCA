import sys
sys.path.append('../')
import SNIDsn
import SNIDdataset as snid
import numpy as np
import SNePCA

import plotly.plotly as ply
import plotly.graph_objs as go
import plotly.tools as tls

import matplotlib.pyplot as plt

import pandas


# ### Load preprocessed SNID datasets 

# datasetX contains the SNID spectra for the phase range X +/- 5 days, where each SNe has only 1 spectrum in this phase range.  The spectrum with phase closest to X is chosen. All of the preprocessing has been applied (wavelength cut, smoothing, phase type, etc)

dataset0 = snid.loadPickle('../../Data/DataProducts/dataset0.pickle')
dataset5 = snid.loadPickle('../../Data/DataProducts/dataset5.pickle')
dataset10 = snid.loadPickle('../../Data/DataProducts/dataset10.pickle')
dataset15 = snid.loadPickle('../../Data/DataProducts/dataset15.pickle')


# ### Run PCA

snidPCA0 = SNePCA.SNePCA(dataset0, -5, 5)
snidPCA5 = SNePCA.SNePCA(dataset5, 0, 10)
snidPCA10 = SNePCA.SNePCA(dataset10, 5, 15)
snidPCA15 = SNePCA.SNePCA(dataset15, 10, 20)

snidPCA0.snidPCA()
snidPCA5.snidPCA()
snidPCA10.snidPCA()
snidPCA15.snidPCA()


# Choose the arbitrary signs for the eigenspectra so that they are consistent across phases, and so that the eigenspectra features match H and He absorption features in the mean spectra.

snidPCA10.evecs[0] = -snidPCA10.evecs[0]

snidPCA5.evecs[1] = -snidPCA5.evecs[1]
snidPCA10.evecs[1] = -snidPCA10.evecs[1]
snidPCA15.evecs[1] = -snidPCA15.evecs[1]

snidPCA0.evecs[2] = -snidPCA0.evecs[2]
snidPCA5.evecs[2] = -snidPCA5.evecs[2]
snidPCA15.evecs[2] = -snidPCA15.evecs[2]

snidPCA0.evecs[3] = -snidPCA0.evecs[3]
snidPCA5.evecs[3] = -snidPCA5.evecs[3]
snidPCA10.evecs[3] = -snidPCA10.evecs[3]

snidPCA0.evecs[4] = -snidPCA0.evecs[4]

snidPCA0.calcPCACoeffs()
snidPCA5.calcPCACoeffs()
snidPCA10.calcPCACoeffs()
snidPCA15.calcPCACoeffs()


# Set colors for plots
snidPCA0.Ib_color = 'steelblue'
snidPCA5.Ib_color = 'steelblue'
snidPCA10.Ib_color = 'steelblue'
snidPCA15.Ib_color = 'steelblue'
snidPCA0.IIb_color = 'limegreen'
snidPCA5.IIb_color = 'limegreen'
snidPCA10.IIb_color = 'limegreen'
snidPCA15.IIb_color = 'limegreen'
snidPCA0.IcBL_color = 'darkgrey'
snidPCA5.IcBL_color = 'darkgrey'
snidPCA10.IcBL_color = 'darkgrey'
snidPCA15.IcBL_color = 'darkgrey'


snidPCA0.IIb_ellipse_color = 'green'
snidPCA5.IIb_ellipse_color = 'green'
snidPCA10.IIb_ellipse_color = 'green'
snidPCA15.IIb_ellipse_color = 'green'
snidPCA0.IcBL_ellipse_color = 'grey'
snidPCA5.IcBL_ellipse_color = 'grey'
snidPCA10.IcBL_ellipse_color = 'grey'
snidPCA15.IcBL_ellipse_color = 'grey'


# # PC's vs Mean Templates

from scipy.io.idl import readsav
import pylab as pl
import numpy as np
import os

meanspec_path = os.environ['MEANSPEC']


def readtemplate(tp):
    if tp=='IcBL' or tp=='Ic':
        s = readsav(meanspec_path + '/meanspec%s_1specperSN_15_ft.sav'%tp)
    else:
        s = readsav(meanspec_path + '/meanspec%s_1specperSN_15.sav'%tp)
    
    return s


def plotPCs(s, tp, c, ax, eig, ewav, sgn):
    lines = []
    for i,e in enumerate(eig):
        line = ax.plot(ewav, sgn[i]*2*e +5-1.0*i, label="PCA%i"%i,c='k')
        lines.append(line)
        if i:
            ax.fill_between(s.wlog, s.fmean + s.fsdev+ 5-1.0*i,
                            s.fmean - s.fsdev +5-1.0*i, 
                    color = c, alpha = 0.2)
        else:
            ax.fill_between(s.wlog, s.fmean + s.fsdev +5-1.0*i,
                            s.fmean - s.fsdev +5-1.0*i, 
                    color = c, alpha = 0.2, label=tp+' Template')
            
    ax.set_xlim(4000,7000)
    ax.set_xlabel("wavelength ($\AA$)",fontsize=26)
    ax.set_ylim(0, 8)
    return ax, lines

plt.clf()
snIb = readtemplate('Ib')
snIc = readtemplate('Ic')
snIIb = readtemplate('IIb')
snIcBL = readtemplate('IcBL')



import matplotlib.patches as mpatches
plt.clf()
f, axs = plt.subplots(2,2,figsize=(25,20), sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.05, wspace=0.05)
axs[0,0], _ = plotPCs(snIIb, 'IIb',snidPCA15.IIb_color, axs[0,0], snidPCA15.evecs[0:5], snidPCA0.wavelengths,[1,1,1,1,1])
axs[0,1], _ = plotPCs(snIb, 'Ib',snidPCA15.Ib_color, axs[0,1], snidPCA15.evecs[0:5], snidPCA0.wavelengths,[1,1,1,1,1])
axs[1,0], _ = plotPCs(snIcBL, 'IcBL',snidPCA15.IcBL_color, axs[1,0], snidPCA15.evecs[0:5], snidPCA0.wavelengths,[1,1,1,1,1])
axs[1,1], lines = plotPCs(snIc, 'Ic',snidPCA15.Ic_color, axs[1,1], snidPCA15.evecs[0:5], snidPCA0.wavelengths,[1,1,1,1,1])
leg = [el[0] for el in lines]
red_patch = mpatches.Patch(color='steelblue', label='Ib Mean Spec', alpha=0.1)
green_patch = mpatches.Patch(color='limegreen', label='IIb Mean Spec', alpha=0.1)
black_patch = mpatches.Patch(color='darkgrey', label='IcBL Mean Spec', alpha=0.1)
blue_patch = mpatches.Patch(color='r', label='Ic Mean Spec', alpha=0.1)
leg.append(green_patch)
leg.append(black_patch)
leg.append(red_patch)
leg.append(blue_patch)



ymax = 7.8
xmin = 3800

fontsz=35
axs[0,0].set_ylim((0,ymax))
axs[0,0].set_xlabel('')
axs[0,1].set_xlabel('')



ax = axs[0,0]
ax.axvspan(6213, 6366, alpha=0.1, color='k') #H alpha -9000 km/s to -16000 km/s
s = r'$\alpha$'
ax.text((6213+6366)/2.0, ymax + ymax * 0.02, 'H'+s, fontsize=fontsz, horizontalalignment='center')
ax.axvspan(4602, 4715, alpha=0.1, color='k') #H Beta -9000 km/s to-16000 km/s
s = r'$\beta$'
ax.text((4602+4715)/2.0, ymax + ymax * 0.02, 'H'+s, fontsize=fontsz, horizontalalignment='center')
ax.axvspan(5621, 5758, alpha=0.1, color='k') #HeI5876 -6000 km/s to -13000 km/s
ax.text((5621+5758)/2.0, ymax + ymax * 0.02, 'HeI5876', fontsize=fontsz, horizontalalignment='center')

ax = axs[0,1]
ax.axvspan(5621, 5758, alpha=0.1, color='k') #HeI5876 -6000 km/s to -13000 km/s
ax.text((5621+5758)/2.0, ymax + ymax * 0.02, 'HeI5876', fontsize=fontsz, horizontalalignment='center')


axs[0,0].tick_params(axis='both',which='both',labelsize=fontsz, length=20,direction='inout')
axs[0,0].get_yaxis().set_ticks([])
axs[0,1].tick_params(axis='both',which='both',labelsize=fontsz, length=20,direction='inout')
axs[1,0].tick_params(axis='both',which='both',labelsize=fontsz, length=20,direction='inout')
axs[1,0].get_yaxis().set_ticks([])
axs[1,1].tick_params(axis='both',which='both',labelsize=fontsz, length=20,direction='inout')


axs[0,0].set_ylim((0,ymax))
axs[0,0].set_xlim((xmin, 7100))
axs[1,0].set_xlim((xmin,7100))
axs[1,0].set_xlabel('')
axs[1,1].set_xlabel('')


f.text(0.085, 2.0/4.0, 'Relative Flux', verticalalignment='center', rotation='vertical', fontsize=fontsz)
xmax = axs[0,0].get_xlim()[1]

f.text(0.5,0.07, 'Wavelength $(\AA)$', horizontalalignment='center', fontsize=fontsz)

for ax in axs.flatten():
    ax.text(xmin, 5.15, 'PC1', fontsize=50, color='k')
    ax.text(xmin, 4.15, 'PC2', fontsize=50, color='k')
    ax.text(xmin, 3.15, 'PC3', fontsize=50, color='k')
    ax.text(xmin, 2.15, 'PC4', fontsize=50, color='k')
    ax.text(xmin, 1.15, 'PC5', fontsize=50, color='k')
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.tick_params(axis='x', which='minor', direction='inout', length=15)
    
axs[0,0].text(xmax-50, ymax-1.7, 'IIb Mean Spectrum $\pm 1\sigma$', fontsize=50, color='limegreen', ha='right')
axs[0,1].text(xmax-50, ymax-1.7, 'Ib Mean Spectrum $\pm 1\sigma$', fontsize=50, color='steelblue', ha='right')
axs[1,0].text(xmax-50, ymax-1.7, 'IcBL Mean Spectrum $\pm 1\sigma$', fontsize=50, color='darkgrey', ha='right')
axs[1,1].text(xmax-50, ymax-1.7, 'Ic Mean Spectrum $\pm 1\sigma$', fontsize=50, color='r', ha='right')



axs[0,0].text(4000,7.2,'$t_{V_{max}}=15\pm5$',fontsize=35)



f.savefig('Fig3')





