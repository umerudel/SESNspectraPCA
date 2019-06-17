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


# # PC Time Evolution

# The following cells construct a plot that shows the time evolution of the eigenspectra as phase changes.


f_all, axs = plt.subplots(2,1,figsize=(15,15),gridspec_kw={'hspace':0})


ax = axs[0]
ax.set_xlim((4000,7000))
ax.set_ylim((-.2,.35))

l1=ax.plot(snidPCA0.wavelengths,snidPCA0.evecs[4]+0.0,color='k',linewidth=4.0, label='PC5, $t_{V_{max}}=0\pm5$')
l2=ax.plot(snidPCA5.wavelengths, snidPCA5.evecs[2]+0.0,'--',color='r',linewidth=4.0, label='PC3, $t_{V_{max}}=5\pm5$')
l3=ax.plot(snidPCA5.wavelengths, snidPCA10.evecs[2]+0.0,'--',color='green',linewidth=4.0, label='PC3, $t_{V_{max}}=10\pm5$')
l4=ax.plot(snidPCA5.wavelengths, snidPCA15.evecs[2]+0.0,'--',color='b',linewidth=4.0, label='PC3, $t_{V_{max}}=15\pm5$')

ax.legend(handles=[l1[0],l2[0],l3[0],l4[0]],fontsize=30, ncol=2)
ax.tick_params(axis='both',which='major', length=20,direction='inout',labelsize=35)
ax.tick_params(axis='both',which='minor', length=10,direction='inout')
ax.set_yticks([])
ax.set_yticklabels([])
ax.set_xticklabels([])


ax = axs[1]
ax.set_xlim((4000,7000))
ax.set_ylim((-.2,.35))

l1=ax.plot(snidPCA0.wavelengths,snidPCA0.evecs[2]+0.0,color='k',linewidth=4.0, label='PC3, $t_{V_{max}}=0\pm5$')
l2=ax.plot(snidPCA5.wavelengths, snidPCA5.evecs[3]+0.0,'--',color='r',linewidth=4.0, label='PC4, $t_{V_{max}}=5\pm5$')
l3=ax.plot(snidPCA5.wavelengths, snidPCA10.evecs[3]+0.0,'--',color='green',linewidth=4.0, label='PC4, $t_{V_{max}}=10\pm5$')
l4=ax.plot(snidPCA5.wavelengths, snidPCA15.evecs[3]+0.0,'--',color='b',linewidth=4.0, label='PC4, $t_{V_{max}}=15\pm5$')

ax.legend(handles=[l1[0],l2[0],l3[0],l4[0]],fontsize=30,ncol=2)
ax.set_xlabel('Wavelength ($\AA$)',fontsize=50)
ax.tick_params(axis='both',which='major', length=20,direction='inout',labelsize=35)
ax.tick_params(axis='x',which='minor', length=10,direction='inout')
ax.set_yticks([])
ax.set_yticklabels([])



f_all.savefig('Fig4')


