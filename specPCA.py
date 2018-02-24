from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
import numpy as np

NKEEP = 10
    # ## PCA Decomposition
def PCAdecompspec(Ib_Ic_names, spectra_matrix, SN_type_vec,
                  phase_arr, nshow):
    
    msk = np.logical_not(np.in1d(Ib_Ic_names,
                                 np.array(['sn04aw_bsnip','sn2006jc'])))
    spectra_matrix = spectra_matrix[msk]
    Ib_Ic_names = Ib_Ic_names[msk]
    SN_type_vec = SN_type_vec[msk]
    phase_arr = phase_arr[msk]

    pca = PCA()
    pca.fit(spectra_matrix)

    evecs = pca.components_
    evals = pca.explained_variance_ratio_
    evals_cs = evals.cumsum()
    
    n_evecs = evecs.shape[0]
    n_feat = evecs.shape[1]

    print ("eigen vectors", evecs.shape)
    print ("eigen values", evals.shape)

    f = plt.figure(figsize=(15,20))
    if nshow is None:
        nshow = NKEEP
        
    for i,ev in enumerate(evecs[:nshow]):
        #plt.figure(figsize=(15,2))
        plt.subplot(nshow,1,i+1)
        plt.plot(wavelengths,evals[i]*(ev), label="component: %d, %.2f"%(i,evals_cs[i]))
        if i==0:
            plt.title('PCA Eigenspectra Phase 15$\pm$5',fontsize=18)
        if i==nshow-1:
            plt.xlabel('Wavelength',fontsize=16)
        plt.ylabel('Rel Flux',fontsize=16)
        plt.legend(fontsize=12)
    
        plt.show()

    old_evecs = np.copy(evecs)
    old_evals = np.copy(evals)

    old_evals_cs = old_evals.cumsum()

    oldrow = np.copy(old_evecs[1])
    oldnum = np.copy(old_evals[1])
    old_evecs[1] = old_evecs[3]
    old_evals[1] = old_evals[3]
    old_evecs[3] = oldrow
    old_evals[3] = oldnum
    old_evals_cs = old_evals.cumsum()

    if nshow == 8:
        f = plt.figure(figsize=(15,20))
        sign = [1,1,1,-1,-1,1,1,1]
    
        for i,ev in enumerate(evecs[:nshow]):
            #plt.figure(figsize=(15,2))
            plt.subplot(nshow,1,i+1)
            plt.plot(wavelengths,old_evals[i]*sign[i]*(old_evecs[i]), color='r',label="ph 15 component: %d, %.2f"%(i,old_evals_cs[i]))
            plt.plot(wavelengths,evals[i]*(ev),color='b', label="ph 0 component: %d, %.2f"%(i,evals_cs[i]))
            if i==0:
                plt.title('PCA Eigenspectra Comparison Phase $15\pm5$ vs Phase $0\pm5$',fontsize=18)
            if i==nshow-1:
                plt.xlabel('Wavelength',fontsize=16)
            plt.ylabel('Rel Flux',fontsize=16)
            plt.legend(fontsize=10,loc=4)
    
        plt.savefig("plots/eigenspectra_phase_comparison_15_vs_0.png")
    return (evecs)

