# DataProducts

DatasetX.pickle contains the preprocessed set of spectra at phase = X +/- 5 days relative to the date of V-band maximum. Each dataset is a pickled SNIDdataset object, and each SNIDdataset object is a dictionary of SNIDsn objects. The SNIDdataset.py and SNIDsn.py files are located in the /code directory. <b>/svm_score_tables</b> is a directory that contains pickled pandas tables storing the svm mean scores and standard deviations from [Williamson & Modjaz & Bianco (2019)](https://arxiv.org/abs/1903.06815).
