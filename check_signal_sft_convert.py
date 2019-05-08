#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import os
from sh import gunzip #note: sh is not part of the standard install


# In[88]:


def sft(Signal, FS):
    #this function plots the spectrum of a signal
    #Signal: a vector
    #FS: sampling frequency in Hz
    
    #fft
    #X_mags = fft(Signal, N)
    
    #N=len(Signal)
    #bin_vals = np.arange(N)
    #fax_Hz = bin_vals*FS/N
    
    #N_2 = int(math.ceil(N/2))
    
    #fax_Hz = fax_Hz[0:N_2+1]
    #X_mags = X_mags[0:N_2+1]
    
    #absX_mags= [abs(x) for x in X_mags]
    #fig, ax = plt.subplots()
    #ax.plot(fax_Hz, absX_mags)

    #ax.set(xlabel='Frequency (Hz)', ylabel='Magnitude')
    #plt.show()
    
    Fs = FS;                      #Sampling frequency
    T = 1.0/Fs;                   #Sample time
    L = len(Signal);              #Length of signal
    t = np.arange(L)*T            #Time vector
    y = Signal-np.mean(Signal);
    
    NFFT = 10*2**math.ceil(np.log2(abs(L))) #Next power of 2 from length of y
    FT = fft(y,int(NFFT))/L;
    FT = FT [1:int(NFFT)/2+2];
    f = Fs/2*np.linspace(0,1,int(NFFT)/2+1);

    #Plot single-sided amplitude spectrum.
    #absFT= [abs(x) for x in FT]
    #fig1, ax1 = plt.subplots()
    #ax1.plot(f, absFT)
    #plt.xlim(0.01, 1.2) #Check index, Matlab code:xlim ([0.01 1.2]);
    #Matlab code: ylim ([0 max(abs(FT(f>0.01)))]); -> need to do f>0.01
    #absFT = [abs(x) for x > 0.01 in FT] #doesn't work
    #plt.ylim(0, max(absFT));
    #ax1.set(xlabel='Frequency (Hz)', ylabel='|Y(f)|',
    #   title='Single-Sided Amplitude Spectrum of y(t)')
    
    return [f, FT]


# In[ ]:


# Generates figures to compare raw unnormalized 4D to processed unnormalized 4D. Called by fsl_preprocessing_pipeline()

# input: string pathway to subject ('.../subjectlink/fmri/Resting')
# nifti1: name of unnormalized nifti ('unnormalized_no_physio')
# nifti2: 4D matrix of processed nifti (see fsl_preprocessing_pipeline)

# f: 4x2 figure with mean volume signal, volume-wise spatial variance, slice-wise spatial variance by volume, voxelwise temporal variance by slice 

# -- Heidi Jiang, 8/2012
# Modified by Hesam Jahanian, 10/2014
# Converted to Python by Hannah Redden, 5/2019

def check_signal(input, nifti1, nifti2, TR):
#addpath(genpath('/home/shirer/NIFTIUtils/')); #still in Matlab...

inputnifti = os.path.join(input,nifti1)

for x in [1, 2]:
    if i==1: #need to check contents
        filename = inputnifti + 'nii.gz'
        if os.path.exists(filename):
            #Matlab code:gunzip(filename)
            gunzip(filename)
        input_nii = cbiReadNifti(filename) #cbiReadNifti -> need to convert to python too?
        [X,Y,Z,T] = np.shape(input_nii)
        #Matlab code: input_nii = input_nii(:,:,:,4:end);
        input_nii = input_nii[:,:,:,3:T] #check indicies
        os.remove(inputnifti + '.nii')
    if i==2:
        input_nii = nifti2
        del nifti2 #check to make sure del is the same as matlab's clear

    [X,Y,Z,T] = np.shape(input_nii)
    
    input_nii = np.reshape(input_nii, (X*Y*Z,T))
    #Need to check... array versus element?
    #Matlab code: input_nii(input_nii==0)=np.nan
    input_nii = np.where(input_nii==0, np.nan, input_nii) 

    vol_mean = np.nanmean(input_nii) #need to check
    vol_var = np.transpose(np.nanvar(input_nii,1)) #need to check

    [freq FFT]=sft(vol_mean-mean(vol_mean),1/TR) #might need to change to 1.0
    
    input_nii = np.reshape(input_nii, (X*Y,Z,T))
    
    zslice_var = np.squeeze(np.nanvar(input_nii,1)) #need to check
    tslice_var = np.squeeze(np.nanvar(input_nii,[],3)) #need to check
    
    del input_nii
    
    if i==2: #need to check
        #there's probably a better way to write this
        #Matlab code:zslice_var(1:10,:)=np.nan
        dimsz = np.shape(zslice_var)
        for x in np.arange(10):
            tempz = zslice_var[x]
            for i in np.arange(dimsz[1]):
                tempz[i]=np.nan;

    total_mean = np.mean(vol_mean)
    #vol_mean = [x/total_mean for x in vol_mean]
    vol_mean = np.divide(vol_mean, np.mean(total_mean))
    if i ==1:
        #f = figure; #still in matlab
        f.set_visible(not f.get_visible())
        #Need to check, Matlab code: f = figure('visible','off');
    

    plt.subplot(5,2,i) #CHECK INDEX
    fig1, ax1 = plt.subplots()
    ax1.plot(vol_mean)
    ax1.grid()
    ax1.set(xlabel='Volume', ylabel='Mean', title='Scaled volume to volume mean')
    
    plt.subplot(5,2,(i+2)) #CHECK INDEX
    absFFT= [abs(x) for x in FFT]
    fig2, ax2 = plt.subplots()
    ax2.plot(freq, absFFT)
    ax2.grid()
    plt.xlim(0.01, 1.2*0.35/TR) #Check index, Matlab code:xlim ([0.01 1.2*0.35/TR]);
    #Matlab code: ylim ([0 max(abs(FT(freq>0.01)))]);
    #is this right? not sure what the matlab code is doing
    for a in np.shape(freq): #need to check -> element vs array, need FT[a] as an element
        if freq[a]>0.01:
            FT[a] = abs(FT(a))
    plt.ylim(0, max(FT));
    ax2.set(xlabel='Frequency (Hz)', ylabel='Magnitude', title='FFT')

    diff_vol_var = np.diff(vol_var)
    absDiff_vol_var = [abs(x) for x in diff_vol_var]
    #vol_var = [x/total_mean for x in absDiff_vol_var]
    vol_var = np.divide(diff_vol_var, total_mean)

    del reshaped_input

    plt.subplot(5,2,(i+4)) #CHECK INDEX
    fig3, ax3 = plt.subplots()
    ax3.plot(vol_var)
    ax3.grid()
    ax3.set(xlabel='Difference Volume', ylabel='Spatial Variance',
       title='Scaled volume to volume variance')

    #zslice_var = np.transpose([x/total_mean for x in zslice_var])
    zslice_var = np.transpose(np.divide(zslice_var, total_mean))
    diff_zslice_var = np.diff(zslice_var)
    zslice_var = [abs(x) for x in diff_zslice_var]

    #tslice_var = np.transpose([x/total_mean for x in tslice_var])
    tslice_var = np.transpose(np.divide(tslice_var, total_mean))

    if i==2: #need to check
        #there's probably a better way to write this
        #Matlab code:tslice_var(1:10,:)=np.nan
        dimst = np.shape(tslice_var)
        for x in np.arange(10):
            temp = tslice_var[x]
            for i in np.arange(dimst[1]):
                temp[i]=np.nan;

    plt.subplot(5,2,(i+6)) #CHECK INDEX
    fig4, ax4 = plt.subplots()
    ax4.plot(zslice_var, 'x')
    ax4.grid()
    ax4.set(xlabel='Difference Volume', ylabel='Spatial Variance',
       title='Scaled variance by slice (omit 10 inferior-most)')

    plt.subplot(5,2,(i+8)) #CHECK INDEX
    fig5, ax5 = plt.subplots()
    ax5.plot(tslice_var, 'x')
    ax5.grid()
    ax5.set(xlabel='Axial Slice', ylabel='Temporal Variance',
       title='Scaled voxelwise temporal variance by slice')
   
    del zslice_var
    del diff_zslice_var
    del tslice_var


# In[ ]:
