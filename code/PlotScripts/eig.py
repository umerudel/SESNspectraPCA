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



# # Eigenspectra

# Each panel of the following plot shows one of the first 5 eigenspectra in every phase range.  The eigenspectra signs are consistent across phases.

# In[15]:

f, axs = plt.subplots(5,1,figsize=(10,20))


# In[16]:

for i in range(5):
    ax = axs[i]
    ax.set_title('PCA%d Eigenspectra'%(i+1))
    ax.set_xlim((3800,7700))
    ax.set_yticks([])
    ax.plot(snidPCA0.wavelengths,snidPCA0.evecs[i]+2-0,'r',label='phase 0')
    ax.plot(snidPCA0.wavelengths,snidPCA5.evecs[i]+2-.5,'b',label='phase 5')
    ax.plot(snidPCA0.wavelengths,snidPCA10.evecs[i]+2-1,'c',label='phase 10')
    ax.plot(snidPCA0.wavelengths,snidPCA15.evecs[i]+2-1.5,'g',label='phase 15')
    
    #ax.plot(snidPCA20.wavelengths,snidPCA20.evecs[i]+2-2,'k',label='phase 20')
    #ax.plot(snidPCA25.wavelengths,snidPCA25.evecs[i]+2-2.5,'y',label='phase 25')
    
    ax.legend()


# In[17]:

f.savefig('eigenspectra_all_phases')

