#!/usr/bin/env python3
import sys
from sys import argv
import SNIDsn
import SNIDdataset as snid
import numpy as np
import SNePCA

import plotly.plotly as ply
import plotly.graph_objs as go
import plotly.tools as tls

import matplotlib.pyplot as plt

import pandas
from BinSpectra import lowres_dataset
import warnings

PATH = '../Data/DataProducts/'

# dataset0 = snid.loadPickle('../Data/DataProducts/dataset0.pickle')
# dataset5 = snid.loadPickle('../Data/DataProducts/dataset5.pickle')
# dataset10 = snid.loadPickle('../Data/DataProducts/dataset10.pickle')
# dataset15 = snid.loadPickle('../Data/DataProducts/dataset15.pickle')
def loaddata(phase):
	dsname = "dataset{}.pickle".format(phase)
	return snid.loadPickle(PATH + dsname)


def classify_spectra(ph, bin_length, dphase=5):
    warnings.filterwarnings('ignore')
    datain = loaddata(phase)
    dataset_lowres = lowres_dataset(datain, bin_length)
    snidPCA = SNePCA.SNePCA(dataset_lowres, ph - dphase, ph+phase)
    snidPCA.snidPCA()
    snidPCA.calcPCACoeffs()
    svm_score_dict = {}
    f_all, axs = plt.subplots(5,2,figsize=(35,70),gridspec_kw={'wspace':.2,'hspace':.2})
    #replace with double for loop
    l = 0
    while l < 9:
        for i in range(1, 5):
            for j in range(i + 1, 6):
                exclude = ['sn2007uy', 'sn2009er', 'sn2005ek']
                print(k,i,j)
                f_all,svmsc,av,std=snidPCA.pcaPlot(i,j,(10,7),alphamean=.5,alphaell=.1,alphasvm=10,purity=True,excludeSNe=exclude,std_rad=1.0,svm=True,count=3,fig=f_all,ax=f_all.axes[l],ncv=50,markOutliers=True)
                leg = f_all.axes[l].get_legend()
                tit = leg.get_title()
                leg.set_title(title=None)
                xmin, xmax = f_all.axes[l].get_xlim()
                ymin, ymax = f_all.axes[l].get_ylim()
                f_all.axes[l].tick_params(axis='both',which='major', length=20,direction='inout',labelsize=35)
                f_all.axes[l].tick_params(axis='both',which='minor', length=10,direction='inout')
                f_all.axes[l].set_ylabel('PC%d'%j, fontsize=50)
                f_all.axes[l].set_xlabel('PC%d'%i, fontsize=50)
                f_all.axes[l].text(xmin + .3,ymax - .5,'SVM Test Score = %.2f $\pm$ %.2f'%(av,std),fontsize=40)
                svm_score_dict['PC%d vs PC%d'%(i, j)] = 'SVM Test Score = %.2f \u00B1 %.2f'%(av, std)
                svm_score_dict.update(svm_score_dict)
                l = l + 1
    return svm_score_dict, f_all

if __name__ == '__main__':
    phase = sys.argv[1]
    #fix it
    assert (phase >= int(phase)), "argument 1 should be an integer"
    bin_length = int(sys.argv[2])
    if len(sys.argv) > 3:
	    dphase = int(sys.argv[3])
        classify_spectra(phase, bin_length, dphase)
    else:
	classify_spectra(phase, bin_length)
