# SESNspectraPCA

This repository contains the code for running the PCA + SVM spectral analysis on Stripped-Envelope Core-Collapse Supernovae found in Williamson et al. (2019). There are also data products associated with the new SNID templates that were added to the SESNe spectral library compiled from [Liu et al. 2016](http://adsabs.harvard.edu/abs/2016ApJ...827...90L) and [Modjaz et al. 2016](http://adsabs.harvard.edu/abs/2016ApJ...832..108M). The repository is organized into the following folders:
- <b>/PCvsTemplates</b> contains mean spectra for the phase 15 days presented in [Liu et al. 2016](http://adsabs.harvard.edu/abs/2016ApJ...827...90L). These are used to compare the mean spectra with the eigenspectra calculated by our PCA analysis.
- <b>/SNePCAplots</b> contains final figures showing PCA reconstruction of SN2011ei, cumulative variance captured by principal components, time evolution of the eigenspectra, comparison of mean spectra vs eigenspectra, and SVM classification of SESNe.
- <b>/allSNIDtemp</b> contains the SNID templates (\*.lnw files) used in Williamson et al. 2019.
- <b>/old_code and /old_notebooks </b> contain unused or old versions of code files and notebooks.
- <b>/code </b> contains the SNePCA.py which implements the PCA and SVM analysis used in Williamson et al. 2019. It also contains SNIDsn.py and SNIDdataset.py which are used to load the SNID template files (\*.lnw files) and compile them into a dataset for easy use. There is a tutorial jupyter notebook, <b> Tutorial.ipynb </b> which generates plots from Williamson et al. (2019) and demonstrates how the code is run.

### Acknowledgement:

If you use data products or the analysis in this code, please <b>acknowledge</b> this work by citing in your paper:  Williamson et al. 2019.

Williamson & Modjaz & Bianco (2019):

  	@misc{1903.06815,
    Author = {Marc Williamson and Maryam Modjaz and Federica Bianco},
    Title = {Optimal Classification and Outlier Detection for Stripped-Envelope Core-Collapse Supernovae},
    Year = {2019},
    Eprint = {arXiv:1903.06815},
    }
