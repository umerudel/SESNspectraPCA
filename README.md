# SESNspectraPCA

This repository contains the code for running the PCA + SVM spectral analysis on Stripped-Envelope Core-Collapse Supernovae found in Williamson et al. (2019). There are also data products associated with the new SNID templates that were added to the SESNe spectral library compiled from [Liu et al. 2016](http://adsabs.harvard.edu/abs/2016ApJ...827...90L) and [Modjaz et al. 2016](http://adsabs.harvard.edu/abs/2016ApJ...832..108M). The repository is organized into the following folders:
- <b>/Data contains the pickled spectral datasets used in Williamson et al. (2019) at phases 0, 5, 10, 15 days relative to V-band maximum. The spectra have been preprocessed. These pickled files are used in the Tutorial notebook.</b>
- <b>/PCvsTemplates</b> contains mean spectra for the phase 15 days presented in [Liu et al. 2016](http://adsabs.harvard.edu/abs/2016ApJ...827...90L). These are used to compare the mean spectra with the eigenspectra calculated by our PCA analysis.
- <b>/SNePCAplots</b> contains final figures showing PCA reconstruction of SN2011ei, cumulative variance captured by principal components, time evolution of the eigenspectra, comparison of mean spectra vs eigenspectra, and SVM classification of SESNe.
- <b>/code </b> contains the SNePCA.py which implements the PCA and SVM analysis used in Williamson et al. 2019. It also contains SNIDsn.py and SNIDdataset.py which are used to load the SNID template files (\*.lnw files) and compile them into a dataset for easy use. There is a tutorial jupyter notebook, <b> Tutorial.ipynb </b> which generates plots from Williamson et al. (2019) and demonstrates how the code is run.

# Compiling the SESNe SNID Dataset

In order to rerun the entirety of our analysis, including preprocessing, it is necessary to compile the dataset of SuperNova IDentification (SNID; [Blondin & Tonry 2007](https://iopscience.iop.org/article/10.1086/520494/meta)) Templates used in Williamson et al. (2019). Follow these steps:

1. <b>Download the default SNID library</b> -- The default library of SESNe spectra for SNID can be found on Stephan Blondin's [website](https://people.lam.fr/blondin.stephane/software/snid/)
2. <b>Add SNID Templates from [nyusngroup](https://github.com/nyusngroup/SESNtemple)</b> 

### Acknowledgement:

If you use data products or the analysis in this code, please <b>acknowledge</b> this work by citing in your paper:  Williamson et al. 2019.

Williamson & Modjaz & Bianco (2019):

  	@article{1903.06815,
    Author = {Marc Williamson and Maryam Modjaz and Federica Bianco},
    Title = {Optimal Classification and Outlier Detection for Stripped-Envelope Core-Collapse Supernovae},
    Year = {2019},
    Eprint = {arXiv:1903.06815},
    }
