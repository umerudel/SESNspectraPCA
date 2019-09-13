#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from BinSpectra import binspec_left
from BinSpectra import binspec_right
from BinSpectra import mybinspec
from SNIDsn import binspec


# In[2]:


def testbin(wvl, flux, bin_factor, right_edge_drop_len, left_edge_drop_len):
    flux_edge = flux
    flux_edge[0:left_edge_drop_len] = 0
    flux_edge[-right_edge_drop_len:-1] = 0
    flux_edge[-1] = 0
    new_spectrum_left = binspec_left(wvl, flux_edge, flux, bin_factor)
    new_spectrum_right = binspec_right(wvl, flux_edge, flux, bin_factor)
    fig, axs = plt.subplots(2, figsize=(12,12))
    #fig.suptitle('', fontsize=20)
    axs[0].plot(wvl, flux, linewidth=2)
    axs[0].scatter(wvl, flux,  s=50)
    axs[0].plot(new_spectrum_left[0], new_spectrum_left[1], linewidth=2)
    axs[0].scatter(new_spectrum_left[0], new_spectrum_left[1],  s=50)
    axs[1].plot(wvl, flux, 'tab:red', linewidth=2)
    axs[1].scatter(wvl, flux, s=50, c='tab:red')
    axs[1].plot(new_spectrum_right[0], new_spectrum_right[1], 'tab:green', linewidth=2)
    axs[1].scatter(new_spectrum_right[0], new_spectrum_right[1],  s=50, c='tab:green')
    axs[1].set_xlabel('Wavelength', fontsize=20)
    axs[0].set_ylabel('Flux', fontsize='20')
    axs[1].set_ylabel('Flux', fontsize='20')
    axs[0].tick_params(axis='both', which='major', labelsize=20)
    axs[1].tick_params(axis='both', which='major', labelsize=20)
    #fig.savefig('plot.pdf')
    #print("No. of new bins should be:", int(float(len(wvl)) / bin_factor),  ";  No. of new bins are:", new_spectrum[0].shape)
    #plt.plot(wvl, flux, label="Original resolution")
    #plt.scatter(wvl, flux, label="Original resolution")
    #plt.plot(new_spectrum[0], new_spectrum[1], label="Binning factor: bin_factor")
    #plt.scatter(new_spectrum[0], new_spectrum[1], label="Binning factor: bin_factor")
    #plt.xlabel('Wavelength', fontsize=14)
    #plt.ylabel('Flux', fontsize=14)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    return


def compare_binspec(wvl, flux, wbin, sn_name):
    ns_mybinspec = mybinspec(wvl, flux, flux, wbin)
    wstart = ns_mybinspec[0][0]
    wend = ns_mybinspec[0][-1]
    ns_binspec = binspec(wvl, flux, wstart, wend, wbin)
    fig, axs = plt.subplots(3, figsize=(12,12))
    fig.suptitle('Bin length = %.2f'%(wbin), fontsize=20)
    axs[0].plot(wvl, flux, linewidth=2, label = "Original")
    #axs[0].scatter(wvl, flux,  s=50)
    axs[1].plot(ns_binspec[1], ns_binspec[0], 'tab:red', linewidth=2, label = "Marc's binspec")
    #axs[0].scatter(new_spectrum_left[0], new_spectrum_left[1],  s=50)
    #axs[1].plot(wvl, flux, 'tab:red', linewidth=2)
    #axs[1].scatter(wvl, flux, s=50, c='tab:red')
    axs[2].plot(ns_mybinspec[0], ns_mybinspec[1], 'tab:green', linewidth=2, label = "My binspec")
    #axs[1].scatter(new_spectrum_right[0], new_spectrum_right[1],  s=50, c='tab:green')
    axs[2].set_xlabel('Wavelength', fontsize=20)
    axs[0].set_ylabel('Flux', fontsize='20')
    axs[1].set_ylabel('Flux', fontsize='20')
    axs[2].set_ylabel('Flux', fontsize='20')
    axs[0].tick_params(axis='both', which='major', labelsize=20)
    axs[1].tick_params(axis='both', which='major', labelsize=20)
    axs[2].tick_params(axis='both', which='major', labelsize=20)
    axs[0].legend(loc="upper right", fontsize = '12')
    axs[1].legend(loc="upper right", fontsize = '12')
    axs[2].legend(loc="upper right", fontsize = '12')
    #axs[0].text(5000,0.55, textstr, fontsize=20)
    #fig.savefig('plot.pdf')
    #print("No. of new bins should be:", int(float(len(wvl)) / bin_factor),  ";  No. of new bins are:", new_spectrum[0].shape)
    #plt.plot(wvl, flux, label="Original resolution")
    #plt.scatter(wvl, flux, label="Original resolution")
    #plt.plot(new_spectrum[0], new_spectrum[1], label="Binning factor: bin_factor")
    #plt.scatter(new_spectrum[0], new_spectrum[1], label="Binning factor: bin_factor")
    #plt.xlabel('Wavelength', fontsize=14)
    #plt.ylabel('Flux', fontsize=14)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    return

