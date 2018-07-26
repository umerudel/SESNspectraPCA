from __future__ import division

import SNIDsn
import SNIDdataset as snid

import numpy as np
import scipy

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_color_codes('colorblind')
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
import matplotlib.gridspec as gridspec

import plotly.plotly as ply
import plotly.graph_objs as go
import plotly.tools as tls

import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import distance

import pickle




class SNePCA:

    def __init__(self, snidset):
        self.snidset = snidset

        self.IIb_color = 'g'
        self.Ib_color = 'mediumorchid'
        self.Ic_color = 'r'
        self.IcBL_color = 'k'
        self.H_color = 'steelblue'
        self.He_color = 'indianred'

        
        nspec = snid.numSpec(self.snidset)
        snnames = self.snidset.keys()
        tmpobj = self.snidset[snnames[0]]
        nwvlbins = len(tmpobj.wavelengths)
        self.wavelengths = tmpobj.wavelengths

        specMatrix = np.ndarray((nspec, nwvlbins))
        count = 0
        for snname in snnames:
            snobj = self.snidset[snname]
            phasekeys = snobj.getSNCols()
            for phk in phasekeys:
                specMatrix[count,:] = snobj.data[phk]
                count = count + 1

        self.specMatrix = specMatrix

        return

    def snidPCA(self):
        pca = PCA()
        pca.fit(self.specMatrix)
        self.evecs = pca.components_
        self.evals = pca.explained_variance_ratio_
        self.evals_cs = self.evals.cumsum()
        self.pcaCoeffMatrix = np.dot(self.evecs, self.specMatrix.T).T

        for i, snname in enumerate(self.snidset.keys()):
            snobj = self.snidset[snname]
            snobj.pcaCoeffs = self.pcaCoeffMatrix[i,:]
        return

    def reconstructSpectrum(self, snname, phasekey, nPCAComponents, fontsize):
        snobj = self.snidset[snname]
        datasetMean = np.mean(self.specMatrix, axis=0)
        trueSpec = snobj.data[phasekey]
        pcaCoeff = np.dot(self.evecs, (trueSpec - datasetMean))
        f = plt.figure(figsize=(15,20))
        plt.tick_params(axis='both', which='both', bottom='off', top='off',\
                            labelbottom='off', labelsize=40, right='off', left='off', labelleft='off')
        f.subplots_adjust(hspace=0, top=0.95, bottom=0.1, left=0.12, right=0.93)
        
        for i, n in enumerate(nPCAComponents):
            ax = f.add_subplot(411 + i)
            ax.plot(snobj.wavelengths, trueSpec, '-', c='gray')
            ax.plot(snobj.wavelengths, datasetMean + (np.dot(pcaCoeff[:n], self.evecs[:n])), '-k')
            ax.tick_params(axis='both',which='both',labelsize=20)
            if i < len(nPCAComponents) - 1:
                plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off') # labels along the bottom edge are off
            ax.set_ylim(-2,2)

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


    def plotEigenspectra(self, figsize, nshow, ylim=None, fontsize=16):
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

