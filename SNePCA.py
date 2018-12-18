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
from matplotlib.colors import LinearSegmentedColormap

import plotly.plotly as ply
import plotly.graph_objs as go
import plotly.tools as tls

import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import distance
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

import pickle

from scipy.io.idl import readsav
import pylab as pl

def readtemplate(tp):
    if tp=='IcBL':
        s = readsav('PCvsTemplates/meanspec%s_1specperSN_15_ft.sav'%tp)
    else:
        s = readsav('PCvsTemplates/meanspec%s_1specperSN_15.sav'%tp)
    #pl.fill_between(s.wlog, s.fmean + s.fsdev, s.fmean - s.fsdev, 
    #                color = 'k', alpha = 0.5)
    #pl.plot(s.wlog, s.fmean, label="flattened mean %s phase = 15"%tp
    #        , lw=2)
    #pl.ylabel(r"relative flux", fontsize = 18)
    #pl.xlabel(r"Rest Wavelength $\AA$", fontsize = 18)
    #pl.legend(fontsize = 18)
    #pl.show()
    
    return s
def plotPCs(s, tp, c, ax, eig, ewav, sgn):
    #fig = pl.figure(figsize=(5,5))
    lines = []
    for i,e in enumerate(eig):
       # pl.plot(np.linspace(4000,7000,len(e)), e +0.5*i, label="PC%i"%i)
        line = ax.plot(ewav, sgn[i]*2*e +5-1.0*i, label="PCA%i"%i)
        lines.append(line)
        if i:
            ax.fill_between(s.wlog, s.fmean + s.fsdev+ 5-1.0*i,
                            s.fmean - s.fsdev +5-1.0*i, 
                    color = c, alpha = 0.1)
        else:
            ax.fill_between(s.wlog, s.fmean + s.fsdev +5-1.0*i,
                            s.fmean - s.fsdev +5-1.0*i, 
                    color = c, alpha = 0.1, label=tp+' Template')
            
    ax.set_xlim(4000,7000)
    ax.set_xlabel("wavelength ($\AA$)",fontsize=26)
    #ax.ylabel("relative flux")
    ax.set_ylim(0, 8)
    #ax.legend(ncol=3, loc='upper right', fontsize=20)
    return ax, lines
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 3, x.max() + 3
    y_min, y_max = y.min() - 3, y.max() + 3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z,**params)
    return out


class SNePCA:

    def __init__(self, snidset, phasemin, phasemax):
        self.snidset = snidset
        self.phasemin = phasemin
        self.phasemax = phasemax

        self.IIb_color = 'g'
        self.Ib_color = 'mediumorchid'
        self.Ic_color = 'r'
        self.IcBL_color = 'gray'
        self.H_color = 'steelblue'
        self.He_color = 'indianred'

        
        nspec = snid.numSpec(self.snidset)
        snnames = self.snidset.keys()
        tmpobj = self.snidset[snnames[0]]
        nwvlbins = len(tmpobj.wavelengths)
        self.wavelengths = tmpobj.wavelengths

        specMatrix = np.ndarray((nspec, nwvlbins))
        pcaNames = []
        pcaPhases = []
        count = 0
        for snname in snnames:
            snobj = self.snidset[snname]
            phasekeys = snobj.getSNCols()
            for phk in phasekeys:
                specMatrix[count,:] = snobj.data[phk]
                count = count + 1
                pcaNames.append(snname)
                pcaPhases.append(phk)
        self.pcaNames = np.array(pcaNames)
        self.pcaPhases = np.array(pcaPhases)
        self.specMatrix = specMatrix

        return


    def getSNeTypeMasks(self):
        snnames = self.snidset.keys()
        snnames = self.pcaNames
        typeinfo = snid.datasetTypeDict(self.snidset)
        IIblist = typeinfo['IIb']
        Iblist = typeinfo['Ib']
        Iclist = typeinfo['Ic']
        IcBLlist = typeinfo['IcBL']

        IIbmask = np.in1d(snnames, IIblist)
        Ibmask = np.in1d(snnames, Iblist)
        Icmask = np.in1d(snnames, Iclist)
        IcBLmask = np.in1d(snnames, IcBLlist)

        return IIbmask, Ibmask, Icmask, IcBLmask


    def snidPCA(self):
        pca = PCA()
        pca.fit(self.specMatrix)
        self.evecs = pca.components_
        self.evals = pca.explained_variance_ratio_
        self.evals_cs = self.evals.cumsum()
#        self.pcaCoeffMatrix = np.dot(self.evecs, self.specMatrix.T).T
#
#        for i, snname in enumerate(self.snidset.keys()):
#            snobj = self.snidset[snname]
#            snobj.pcaCoeffs = self.pcaCoeffMatrix[i,:]
        return

    def calcPCACoeffs(self):
        self.pcaCoeffMatrix = np.dot(self.evecs, self.specMatrix.T).T

        for i, snname in enumerate(self.snidset.keys()):
            snobj = self.snidset[snname]
            snobj.pcaCoeffs = self.pcaCoeffMatrix[i,:]
        return


    def reconstructSpectrum(self, figsize, snname, phasekey, nPCAComponents, fontsize, leg_fontsize, ylim=(-2,2), dytick=1):
        snobj = self.snidset[snname]
        datasetMean = np.mean(self.specMatrix, axis=0)
        trueSpec = snobj.data[phasekey]
        pcaCoeff = np.dot(self.evecs, (trueSpec - datasetMean))
        f = plt.figure(figsize=figsize)
        plt.tick_params(axis='both', which='both', bottom='off', top='off',\
                            labelbottom='off', labelsize=40, right='off', left='off', labelleft='off')
        f.subplots_adjust(hspace=0, top=0.95, bottom=0.1, left=0.12, right=0.93)
        
        for i, n in enumerate(nPCAComponents):
            ax = f.add_subplot(411 + i)
            ax.plot(snobj.wavelengths, trueSpec, '-', c='gray', label=snname+' True Spectrum')
            ax.plot(snobj.wavelengths, datasetMean + (np.dot(pcaCoeff[:n], self.evecs[:n])), '-k', label=snname + ' Reconstruction')
            ax.tick_params(axis='both',which='both',labelsize=20)
            if i == 0:
                ax.legend(loc='lower left', fontsize=leg_fontsize)
            if i < len(nPCAComponents) - 1:
                plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off') # labels along the bottom edge are off
            ax.set_ylim(ylim)
            #if i == 0:
            #    yticks = np.arange(ylim[0], ylim[-1]+dytick, dytick)
            #else:
            #    yticks = np.arange(ylim[0] - np.sign(ylim[0])*dytick, ylim[-1], dytick)
            yticks = np.arange(ylim[0] - np.sign(ylim[0])*dytick, ylim[-1], dytick)
            ax.set_yticks(yticks)
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=10, direction="inout")

            if i == 0:
                # Balmer lines
                trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                trans2 = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)

                #ax.text(0.02,1.03, "(N PCA, %$\sigma^{2}$)", fontsize=fontsize, horizontalalignment='left',\
                #        verticalalignment='center', transform=trans2)

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
                #HeI 5876, 6678, 7065
                ax.axvspan(5621, 5758, alpha=0.1, color=self.He_color) #HeI5876 -6000 km/s to -13000 km/s
                ax.text((5621+5758)/2.0, 1.05, 'HeI', fontsize=fontsize, horizontalalignment='center',\
                        verticalalignment='center', transform=trans)
                ax.axvspan(6388, 6544, alpha=0.1, color=self.He_color)
                ax.text((6388+6544)/2.0, 1.05, 'HeI', fontsize=fontsize, horizontalalignment='center',\
                        verticalalignment='center', transform=trans)
                ax.axvspan(6729, 6924, alpha=0.1, color=self.He_color)
                ax.text((6729+6924)/2.0, 1.05, 'HeI', fontsize=fontsize, horizontalalignment='center',\
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
                #text = '(n PCA = %i,$\sigma^{2}$ =  %.0f'%(n, 100*self.evals_cs[n-1])+'%)'
            ax.text(0.8, 0.3, 'nPCA = %i'%(n), fontsize=fontsize, ha='left', va='top', transform=ax.transAxes)
            text = '$\sigma^{2}$ = %.2f'%(100*self.evals_cs[n-1])+'%'
            #ax.text(0.02, 0.1, text, fontsize=fontsize, ha='left', va='top', transform=ax.transAxes)
            ax.text(0.8, 0.15, text, fontsize=fontsize,ha='left', va='top', transform=ax.transAxes)
            f.axes[-1].set_xlabel(r'${\rm wavelength\ (\AA)}$',fontsize=fontsize)
            f.axes[-1].set_xticklabels(np.arange(4000, 8000, 500),fontsize=fontsize)
            f.axes[-1].tick_params(axis='x', length=10, direction="inout", labelsize=fontsize)
            f.text(0.07, 2.0/4.0, 'Relative Flux', verticalalignment='center', rotation='vertical', fontsize=fontsize)
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



    def meanTemplateEig(self, figsize):
        snIb = readtemplate('Ib')
        snIc = readtemplate('Ic')
        snIIb = readtemplate('IIb')
        snIcBL = readtemplate('IcBL')
        f, axs = plt.subplots(2,2,figsize=figsize, sharex=True, sharey=True)
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        axs[0,0], _ = plotPCs(snIIb, 'IIb','g', axs[0,0], self.evecs[0:5], self.wavelengths,[1,-1,-1,1,-1])
        axs[0,1], _ = plotPCs(snIb, 'Ib','mediumorchid', axs[0,1], self.evecs[0:5], self.wavelengths,[1,-1,-1,1,-1])
        axs[1,0], _ = plotPCs(snIcBL, 'IcBL','k', axs[1,0], self.evecs[0:5], self.wavelengths,[1,-1,-1,1,-1])
        axs[1,1], lines = plotPCs(snIc, 'Ic','r', axs[1,1], self.evecs[0:5], self.wavelengths,[1,-1,-1,1,-1])
        #leg = [el[0] for el in lines]
        #red_patch = mpatches.Patch(color='b', label='Ib Mean Spec', alpha=0.1)
        #green_patch = mpatches.Patch(color='g', label='IIb Mean Spec', alpha=0.1)
        #black_patch = mpatches.Patch(color='k', label='IcBL Mean Spec', alpha=0.1)
        #blue_patch = mpatches.Patch(color='r', label='Ic Mean Spec', alpha=0.1)
        #leg.append(green_patch)
        #leg.append(black_patch)
        #leg.append(red_patch)
        #leg.append(blue_patch)
        
        #plt.figlegend(labels=['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5','IIb Mean Spec', 'IcBL Mean Spec', 'Ib Mean Spec','Ic Mean Spec'],\
        #              handles=leg, loc=(0.25,.45), ncol=4, fontsize=20)
        #plt.suptitle('Eigenspectra vs SNe Mean Templates',size=20)
        #f.savefig('pca_vs_templates.png')
        #f
        return f, axs



    def pcaPlot(self, pcax, pcay, figsize, purity=False, std_rad=None, svm=False, fig=None, ax=None, count=1, svmsc=[], ncv=10):
        if fig is None:
            f = plt.figure(figsize=figsize)
        else:
            f = fig
        if ax is None:
            ax = plt.gca()
        else:
            ax = ax
        red_patch = mpatches.Patch(color=self.Ic_color, label='Ic')
        cyan_patch = mpatches.Patch(color=self.Ib_color, label='Ib')
        black_patch = mpatches.Patch(color=self.IcBL_color, label='IcBL')
        green_patch = mpatches.Patch(color=self.IIb_color, label='IIb')

        IIbMask, IbMask, IcMask, IcBLMask = self.getSNeTypeMasks()

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
        ax.scatter(IIbxmean, IIbymean, color=self.IIb_color, alpha=0.5/count, s=400, marker='x')
        ax.scatter(Ibxmean, Ibymean, color=self.Ib_color, alpha=0.5/count, s=400, marker='x')
        ax.scatter(Icxmean, Icymean, color=self.Ic_color, alpha=0.5/count, s=400, marker='x')
        ax.scatter(IcBLxmean, IcBLymean, color=self.IcBL_color, alpha=0.5/count, s=400, marker='x')

        if purity:
            ncomp_arr = [pcax, pcay]
            keys, purity_rad_arr = self.purityEllipse(std_rad, ncomp_arr)
            IIbrad = purity_rad_arr[0]
            Ibrad = purity_rad_arr[1]
            IcBLrad = purity_rad_arr[2]
            Icrad = purity_rad_arr[3]

            ellipse_IIb = mpatches.Ellipse((IIbxmean, IIbymean),2*IIbrad[0],2*IIbrad[1], color=self.IIb_color, alpha=0.1/count)
            ellipse_Ib = mpatches.Ellipse((Ibxmean, Ibymean),2*Ibrad[0],2*Ibrad[1], color=self.Ib_color, alpha=0.1/count)
            ellipse_Ic = mpatches.Ellipse((Icxmean, Icymean),2*Icrad[0],2*Icrad[1], color=self.Ic_color, alpha=0.1/count)
            ellipse_IcBL = mpatches.Ellipse((IcBLxmean, IcBLymean),2*IcBLrad[0],2*IcBLrad[1], color=self.IcBL_color, alpha=0.1/count)

            ax.add_patch(ellipse_IIb)
            ax.add_patch(ellipse_Ib)
            ax.add_patch(ellipse_Ic)
            ax.add_patch(ellipse_IcBL)

        #ax.scatter(x[IIbMask], y[IIbMask], color=self.IIb_color, edgecolors='k',alpha=1/count)
        #ax.scatter(x[IbMask], y[IbMask], color=self.Ib_color, edgecolors='k',alpha=1/count)
        #ax.scatter(x[IcMask], y[IcMask], color=self.Ic_color, edgecolors='k',alpha=1/count)
        #ax.scatter(x[IcBLMask], y[IcBLMask], color=self.IcBL_color, edgecolors='k',alpha=1/count)
        #for i, name in enumerate(self.sneNames[IcBLMask]):
        #    plt.text(x[IcBLMask][i], y[IcBLMask][i], name)

        if svm:
            truth = 1*IIbMask + 2*IbMask + 3*IcMask + 4*IcBLMask
            dat = np.column_stack((x,y))
            linsvm = LinearSVC()

            ncv_scores=[]
            for i in range(ncv):
                trainX, testX, trainY, testY = train_test_split(dat, truth, test_size=0.3)
                linsvm.fit(trainX, trainY)
                score = linsvm.score(testX, testY)
                ncv_scores.append(score)
            
            trainX, testX, trainY, testY = train_test_split(dat, truth, test_size=0.3)

            linsvm.fit(trainX, trainY)
            score = linsvm.score(testX, testY)
            mesh_x, mesh_y = make_meshgrid(x, y, h=0.02)

            #colors=[(0,1,0),(.8,.59,.58),(1,0,0),(0,0,0)]
            colors=['g','mediumorchid','r','gray']
            nbins = 4
            cmap_name = 'mymap'
            cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=nbins)
            plot_contours(ax, linsvm, mesh_x, mesh_y, alpha=0.2/count, cmap=cm)
            svmsc.append(score)



        ax.scatter(x[IIbMask], y[IIbMask], color=self.IIb_color, edgecolors='k',s=100,alpha=1/count)
        ax.scatter(x[IbMask], y[IbMask], color=self.Ib_color, edgecolors='k',s=100,alpha=1/count)
        ax.scatter(x[IcMask], y[IcMask], color=self.Ic_color, edgecolors='k',s=100,alpha=1/count)
        ax.scatter(x[IcBLMask], y[IcBLMask], color=self.IcBL_color, edgecolors='k',s=100,alpha=1/count)

        ax.set_xlim((np.min(x)-2,np.max(x)+2))
        ax.set_ylim((np.min(y)-2,np.max(y)+2))

        ax.set_ylabel('PCA Comp %d'%(pcay),fontsize=20)
        ax.set_xlabel('PCA Comp %d'%(pcax), fontsize=20)
#        plt.axis('off')
        #plt.legend(handles=[red_patch, cyan_patch, black_patch, green_patch], fontsize=18)
        if svm:
            avgsc = np.mean(np.array(ncv_scores))
            ax.legend(handles=[red_patch, cyan_patch, black_patch, green_patch],\
                            title='SVM Test Score = %.2f'%(avgsc), loc='upper right', ncol=2,fancybox=True, prop={'size':30},fontsize=30)
        else:
            ax.legend(handles=[red_patch, cyan_patch, black_patch, green_patch], fontsize=18)
        #plt.title('PCA Space Separability of IcBL and IIb SNe (Phase %d$\pm$%d Days)'%(self.loadPhase, self.phaseWidth),fontsize=22)
        ax.minorticks_on()
        ax.tick_params(
                    axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    labelsize=20) # labels along the bottom edge are off

        return f, svmsc



    def purityEllipse(self, std_rad, ncomp_array):
        ncomp_array = np.array(ncomp_array) - 1
        IIbMask, IbMask, IcMask, IcBLMask = self.getSNeTypeMasks()
        maskDict = {'IIb':IIbMask, 'Ib':IbMask, 'IcBL':IcBLMask, 'Ic':IcMask}
        keys = ['IIb', 'Ib', 'IcBL', 'Ic']
        masks = [IIbMask, IbMask, IcBLMask, IcMask]
        purity_rad_arr = []
        for key,msk in zip(keys,masks):
            centroid = np.mean(self.pcaCoeffMatrix[:,ncomp_array][msk], axis=0)
            std = np.std(self.pcaCoeffMatrix[:,ncomp_array][msk], axis=0)
            print 'centroid', centroid
            dist_from_centroid = np.abs(self.pcaCoeffMatrix[:,ncomp_array][msk] - centroid)
            mean_dist_from_centroid = np.mean(dist_from_centroid, axis=0)
            print 'mean dist from centroid: ', mean_dist_from_centroid
            std_dist_all_components = np.std(dist_from_centroid, axis=0)
            print 'std dist from centroid: ', std_dist_all_components
            purity_rad_all = mean_dist_from_centroid + std_rad * std_dist_all_components
            print 'purity rad all components: ', purity_rad_all
            #purity_rad_arr.append(purity_rad_all)
            purity_rad_arr.append(std)


            ellipse_cond = np.sum(np.power((self.pcaCoeffMatrix[:,ncomp_array] - centroid), 2)/\
                                  np.power(purity_rad_all, 2), axis=1)
            print 'ellipse condition: ', ellipse_cond
            purity_msk = ellipse_cond < 1

            print key
            print 'purity radius: ', purity_rad_all
            print '# of SNe within purity ellipse for type '+key+': ',np.sum(purity_msk)
            names_within_purity_rad = self.pcaNames[purity_msk]
            correct_names = self.pcaNames[msk]
            correct_msk = np.isin(names_within_purity_rad, correct_names)
            print '# of correct SNe '+key+': ', np.sum(correct_msk)
        return keys, purity_rad_arr


    def pcaPlotly(self, pcaxind, pcayind, std_rad):
        IIbmask, Ibmask, Icmask, IcBLmask = self.getSNeTypeMasks()
        pcax = self.pcaCoeffMatrix[:,pcaxind - 1]
        pcay = self.pcaCoeffMatrix[:,pcayind - 1]
        col_red = 'rgba(152,0,0,1)'
        col_blue = 'rgba(0,152,152,1)'
        col_green = 'rgba(0,152,0,1)'
        col_black = 'rgba(0,0,0,152)'
        col_purp = 'rgba(186,85,211, 0.8)'

        #np.array([nm+'_'+ph for nm,ph in zip(self.pcaNames, self.pcaPhases)])
        traceIIb=go.Scatter(x=pcax[IIbmask], y=pcay[IIbmask], mode='markers',\
                            marker=dict(size=10, line=dict(width=1), color=col_green, opacity=1), \
                            text=np.array([nm+'_'+ph for nm,ph in zip(self.pcaNames, self.pcaPhases)])[IIbmask], name='IIb')
        
        traceIb=go.Scatter(x=pcax[Ibmask], y=pcay[Ibmask], mode='markers',\
                            marker=dict(size=10, line=dict(width=1), color=col_purp, opacity=1), \
                            text=np.array([nm+'_'+ph for nm,ph in zip(self.pcaNames, self.pcaPhases)])[Ibmask], name='Ib')
        
        traceIc=go.Scatter(x=pcax[Icmask], y=pcay[Icmask], mode='markers',\
                            marker=dict(size=10, line=dict(width=1), color=col_red, opacity=1), \
                            text=np.array([nm+'_'+ph for nm,ph in zip(self.pcaNames, self.pcaPhases)])[Icmask], name='Ic')
        
        traceIcBL=go.Scatter(x=pcax[IcBLmask], y=pcay[IcBLmask], mode='markers',\
                            marker=dict(size=10, line=dict(width=1), color=col_black, opacity=1), \
                            text=np.array([nm+'_'+ph for nm,ph in zip(self.pcaNames, self.pcaPhases)])[IcBLmask], name='IcBL')
        data = [traceIIb, traceIb, traceIc, traceIcBL]


        #centroids
        IIbxmean = np.mean(pcax[IIbmask])
        IIbymean = np.mean(pcay[IIbmask])
        Ibxmean = np.mean(pcax[Ibmask])
        Ibymean = np.mean(pcay[Ibmask])
        Icxmean = np.mean(pcax[Icmask])
        Icymean = np.mean(pcay[Icmask])
        IcBLxmean = np.mean(pcax[IcBLmask])
        IcBLymean = np.mean(pcay[IcBLmask])
        
        keys, purityrad = self.purityEllipse(std_rad, [pcaxind, pcayind])
        IIbradx = purityrad[0][0]
        IIbrady = purityrad[0][1]
        Ibradx = purityrad[1][0]
        Ibrady = purityrad[1][1]
        IcBLradx = purityrad[2][0]
        IcBLrady = purityrad[2][1]
        Icradx = purityrad[3][0]
        Icrady = purityrad[3][1]

        layout = go.Layout(autosize=False,
               width=1000,
               height=700,
               annotations=[
                   dict(
                       x=1.05,
                       y=1.025,
                       showarrow=False,
                       text='Phases: [%.2f, %.2f]'%(self.phasemin, self.phasemax),
                       xref='paper',
                       yref='paper'
                   )],
               xaxis=dict(
                   title='PCA%i'%(pcaxind),
                   titlefont=dict(
                       family='Courier New, monospace',
                       size=30,
                       color='black'
                   ),
               ),
               yaxis=dict(
                   title='PCA%i'%(pcayind),
                   titlefont=dict(
                       family='Courier New, monospace',
                       size=30,
                       color='black'
                   ),
               ), shapes=[
                   {
                       'type': 'circle',
                       'xref': 'x',
                       'yref': 'y',
                       'x0': IIbxmean-IIbradx,
                       'y0': IIbymean - IIbrady,
                       'x1': IIbxmean+IIbradx,
                       'y1': IIbymean + IIbrady,
                       'opacity': 0.2,
                       'fillcolor': col_green,
                       'line': {
                           'color': col_green,
                       },
                   },
               {
                       'type': 'circle',
                       'xref': 'x',
                       'yref': 'y',
                       'x0': Ibxmean - Ibradx,
                       'y0': Ibymean - Ibrady,
                       'x1': Ibxmean + Ibradx,
                       'y1': Ibymean + Ibrady,
                       'opacity': 0.2,
                       'fillcolor': col_purp,
                       'line': {
                           'color': col_purp,
                       },
                   },{
                       'type': 'circle',
                       'xref': 'x',
                       'yref': 'y',
                       'x0': Icxmean - Icradx,
                       'y0': Icymean - Icrady,
                       'x1': Icxmean + Icradx,
                       'y1': Icymean + Icrady,
                       'opacity': 0.2,
                       'fillcolor': col_red,
                       'line': {
                           'color': col_red
                       }
                   },{
                       'type': 'circle',
                       'xref': 'x',
                       'yref': 'y',
                       'x0': IcBLxmean - IcBLradx,
                       'y0': IcBLymean - IcBLrady,
                       'x1': IcBLxmean + IcBLradx,
                       'y1': IcBLymean + IcBLrady,
                       'opacity': 0.2,
                       'fillcolor': col_black,
                       'line': {
                           'color': col_black
                       }
                   }]
           )
        fig = go.Figure(data=data, layout=layout)
        return fig

    def cornerplotPCA(self, ncomp, figsize, svm=False, ncv=1):
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

        IIbMask, IbMask, IcMask, IcBLMask = self.getSNeTypeMasks()
        svm_highscore = 0.0
        svm_x = -1
        svm_y = -1

        f = plt.figure(figsize=figsize)
        for i in range(ncomp):
            for j in range(ncomp):
                if i > j:
                    plotNumber = ncomp * i + j + 1
                    print plotNumber
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

                    if svm:
                        truth = 1*IIbMask + 2*IbMask + 3*IcMask + 4*IcBLMask
                        dat = np.column_stack((y,x))

                        ncv_scores=[]
                        for cvit in range(ncv):
                            trainX, testX, trainY, testY = train_test_split(dat, truth, test_size=0.3)
                            linsvm = LinearSVC()
                            linsvm.fit(trainX, trainY)
                            score = linsvm.score(testX, testY)
                            ncv_scores.append(score)
                        score = np.mean(ncv_scores)

                        #trainX, testX, trainY, testY = train_test_split(dat, truth, test_size=0.3)

                        #linsvm = LinearSVC()
                        #linsvm.fit(trainX, trainY)
                        #score = linsvm.score(testX, testY)
                        if score > svm_highscore:
                            svm_highscore = score
                            svm_x = j+1
                            svm_y = i+1

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
        #plt.text(-3.0,1.3,'Smoothed IcBL PCA Component 2D Marginalizations (Phase %d$\pm$%d Days)'%(self.loadPhase, self.phaseWidth),fontsize=16)
        if svm:
            return f, svm_highscore, svm_x, svm_y
        return f


    def gridSearchTSNE(self, perplexity_arr, exaggeration_arr, learning_arr):
        high_score = 0.0
        high_perp = None
        high_exagg = None
        high_learn = None
        scores = []
        highTSNE = None
        highSVM = None
        highTSNECoeff = None

        IIbmask, Ibmask, Icmask, IcBLmask = self.getSNeTypeMasks()
        truth = 1*IIbmask + 2*Ibmask + 3*Icmask + 4*IcBLmask
        for perp in perplexity_arr:
            for exagg in exaggeration_arr:
                for learn in learning_arr:
                    ts = TSNE(n_components=2, perplexity=perp, early_exaggeration=exagg, learning_rate=learn)
                    tsCoeff = ts.fit_transform(self.pcaCoeffMatrix)


                    trainX, testX, trainY, testY = train_test_split(tsCoeff, truth, test_size=0.3)

                    linsvm = LinearSVC()
                    linsvm.fit(trainX, trainY)
                    score = linsvm.score(testX, testY)
                    scores.append(score)
                    if score > high_score:
                        high_score = score
                        high_perp = perp
                        high_exagg = exagg
                        high_learn = learn
                        highTSNE = ts
                        highTSNECoeff = tsCoeff
                        highSVM = linsvm

        #highTSNE = TSNE(n_components=2, perplexity=high_perp, early_exaggeration=high_exagg,\
        #                learning_rate=high_learn)
        #highCoeff = highTSNE.fit_transform(self.pcaCoeffMatrix)
        f = plt.figure(figsize=(10,10))
        ax = f.gca()
        ax.scatter(highTSNECoeff[:,0][IIbmask], highTSNECoeff[:,1][IIbmask], c=self.IIb_color)
        ax.scatter(highTSNECoeff[:,0][Ibmask], highTSNECoeff[:,1][Ibmask], c=self.Ib_color)
        ax.scatter(highTSNECoeff[:,0][Icmask], highTSNECoeff[:,1][Icmask], c=self.Ic_color)
        ax.scatter(highTSNECoeff[:,0][IcBLmask], highTSNECoeff[:,1][IcBLmask], c=self.IcBL_color)
        ax.set_xlabel('TSNE0', fontsize=24)
        ax.set_ylabel('TSNE1', fontsize=24)

        red_patch = mpatches.Patch(color=self.Ic_color, alpha=1.0, label='Ic')
        cyan_patch = mpatches.Patch(color=self.Ib_color, alpha=1.0, label='Ib')
        black_patch = mpatches.Patch(color=self.IcBL_color, alpha=1.0, label='IcBL')
        green_patch = mpatches.Patch(color=self.IIb_color, alpha=1.0, label='IIb')
        plt.legend(handles=[red_patch, cyan_patch, black_patch, green_patch],\
                            title='SVM Classification \nSVM Testing Score = %.2f'%(high_score), loc='best', fancybox=True, prop={'size':16})


        #linsvm = LinearSVC()
        #linsvm.fit(highCoeff, truth)
        mesh_x, mesh_y = make_meshgrid(highTSNECoeff[:,0], highTSNECoeff[:,1], h=0.02)

        colors=[(0,1,0),(.8,.59,.58),(1,0,0),(0,0,0)]
        nbins = 4
        cmap_name = 'mymap'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=nbins)
        plot_contours(ax, highSVM, mesh_x, mesh_y, alpha=0.2, cmap=cm)
        return highTSNE, f, high_score, scores
