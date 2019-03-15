# SESNspectraPCA

This repository contains the code for running the PCA + SVM spectral analysis on Stripped-Envelope Core-Collapse Supernovae found in Williamson et al. (2019). There are also data products associated with the new SNID templates that were added to the SESNe spectral library compiled from [Liu et al. 2016](http://adsabs.harvard.edu/abs/2016ApJ...827...90L) and [Modjaz et al. 2016](http://adsabs.harvard.edu/abs/2016ApJ...832..108M). The repository is organized into the following folders:
- <b>/PCvsTemplates</b> contains mean spectra for the phase 15 days presented in [Liu et al. 2016](http://adsabs.harvard.edu/abs/2016ApJ...827...90L). These are used to compare the mean spectra with the eigenspectra calculated by our PCA analysis.
- <b>/SNePCAplots</b> contains final figures showing PCA reconstruction of SN2011ei, cumulative variance captured by principal components, time evolution of the eigenspectra, comparison of mean spectra vs eigenspectra, and SVM classification of SESNe.
- <b>/allSNIDtemp</b> contains the SNID templates (\*.lnw files) used in Williamson et al. 2019.
- <b>/old_code/notebooks </b> contains unused or old versions of code files and notebooks.
