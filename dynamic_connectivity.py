#!/usr/bin/env python3
# Displays the dynamic connectivity for an input seed mask and data set

import numpy as np
import nibabel as nib
import argparse
import shutil
import os
from collections import defaultdict
from timeit import default_timer as timer

# credit to Berkeley Psych 214: https://github.com/psych-214-fall-2016/classwork/blob/master/pearson/solution/pearson.py
def pearson_2d(x, Y):
    """ Pearson product-moment correlation of vectors `x` and array `Y`
    Parameters
    ----------
    x : array shape (N,)
        One-dimensional array to correlate with every column of `Y`
    Y : array shape (N, P)
        2D array where we correlate each column of `Y` with `x`.
    Returns
    -------
    r_xy : array shape (P,)
        Pearson product-moment correlation of vectors `x` and the columns of
        `Y`, with one correlation value for every column of `Y`.
    """
    # Mean-center x -> mc_x
    # Mean-center every column of Y -> mc_Y
    # a : Get sum of products of mc_x and every column of mc_Y
    # b : Get sum of products of mc_x on mc_x
    # c : Get sum of products of every column of mc_Y[:, i] on itself
    # return a / (sqrt(b) * sqrt(c))
    mc_x = x - np.mean(x)
    mc_Y = Y - np.mean(Y, axis=0)  # This is numpy broadcasting
    # You could also do the step above with:
    # mc_Y = Y - np.tile(np.mean(Y, axis=0), (len(x), 1))
    a = mc_x.dot(mc_Y)
    b = mc_x.dot(mc_x)
    c = np.sum(mc_Y ** 2, axis=0)
    return a / (np.sqrt(b) * np.sqrt(c))

'''
Param:
    data = path to processed epi
    roi = path to roi mask
Behavior: 
    Returns a np array for subject and seed and original epi data affine
'''
def read_data(data, roi):
    print("Reading images and loading data")
    subject = read_subject(data)
    seed = read_roi(roi)
    affine = get_affine(data)
    return subject, seed, affine

'''
Param:
    roi = path to roi mask
Behavior: 
    Returns a np array for inputted roi
'''
def read_roi(roi):
    seed = nib.load(roi, mmap=False)
    return seed.get_data()

'''
Param:
    data = path to processed epi
Behavior: 
    Returns a np array for inputted data
'''
def read_subject(data):
    subject = nib.load(data, mmap=False)
    return subject.get_data()

'''
Param:
    data = path to processed epi
Behavior: 
    Returns the data's affine
'''
def get_affine(img):
    img = nib.load(img, mmap=False)
    return img.affine

'''
Param:
    roi_list = text file of roi paths
Behavior: 
    Returns a dictionary with file name as key and np array of roi as value
'''
def read_all_roi(roi_list):
    rois = dict();
    with open(roi_list) as roi_paths:
        for roi in roi_paths:
            name = os.path.basename(roi[:-1])
            rois[name[:name.find(".")]] = read_roi(roi.strip())
    return rois

'''
Param:
    subject_list = text file of subject paths
Behavior: 
    Returns a list of np arrays for each subject
'''
def read_all_subject(subject_list):
    subjects = []
    with open(subject_list) as subs:
       for subject in subs:
           subjects.append(read_subject(subject.strip()))
    return subjects

'''
Param:
   subjects = list of np arrays for each subject
   rois = dictionary of roi name and np array for each roi
Behavior: 
   Returns a dictionary with keys as roi names and values as a list
   of means of every subject's time course extracted from the roi
'''
def calc_mean(subjects, rois):
    roi_means = defaultdict(list)
    for name, roi in rois.items():
        nans = 0
        for subject in subjects:
            roi_tc = extract_seed_tc(roi, subject)
            mean = np.nanmean(roi_tc)
            if (not np.isnan(mean)):
                roi_means[name].append(mean)
            else:
                nans += 1
        if nans:
            print("Removed " + str(nans) + " values from " + name + "'s means")
    return roi_means

'''
Param:
    subjects = list of np arrays for each subject
    rois = dictionary of roi name and np array for each roi
Behavior: 
    Returns a dictionary with keys as roi names and values as a list
    of standard deviations of every subject's time course extracted from the roi
'''
def calc_std(subjects, rois):
    roi_means = defaultdict(list)
    for name, roi in rois.items():
        nans = 0
        for subject in subjects:
            roi_tc = extract_seed_tc(roi, subject)
            std = np.nanstd(roi_tc)
            if (not np.isnan(std)):
                roi_means[name].append(std)
            else:
                nans += 1
        if nans:
            print("Removed " + str(nans) + " values from " + name + "'s stds")
    return roi_means

'''
Param:
    window_length = window length in trs
    overlap = overlap in trs
    volumes = total number of volumes
Behavior: 
    Returns the number of windows
'''
def calc_windows(window_length, overlap, volumes):
    return (volumes - window_length) // (window_length - overlap) + 1

# Extract time course of seed region
def extract_seed_tc(mask, data):
    print("Extracting the time course of seed region")
    if (mask.shape != data.shape[:-1]):
        print("WARNING: Make sure mask has same dimensions as data!")
    masked_data = np.array([data[index] for index in zip(*np.where(mask > 0))])
    if (np.isnan(masked_data).any()):
        print ("WARNING: There are NaN values within the mask")
    return np.nanmean(masked_data, axis=0)

# Calculate correlation for every voxel 
def calculate_dynam_corr(windows, window_length, overlap, seed_tc, data):
    print("Calculating correlation")
    voxel_tc = data.reshape((np.prod(data.shape[:-1]), data.shape[-1]))
    correlation = np.zeros((data.shape[0], data.shape[1], data.shape[2], windows))
    for window in range(windows):
        win_lower = window * (window_length - overlap)
        win_upper = win_lower + window_length
        print str(win_lower) + " " + str(win_upper)
        correlation[..., window] = pearson_2d(seed_tc[win_lower:win_upper], voxel_tc[:, win_lower:win_upper].T).reshape(data.shape[:-1])
    return correlation
    
def calculate_static_corr(seed_tc, data):
    print("Calculating correlation")
    voxel_tc = data.reshape((np.prod(data.shape[:-1]), data.shape[-1]))
    return pearson_2d(seed_tc, voxel_tc.T).reshape(data.shape[:-1])
    
# Calculate the z-score from correlation matrix
def corr_to_z(correlation):
    return np.arctanh(correlation)

# Save correlation matrix as Nifti Image
def save_output(out_matrix, header, output_name, output_dir):
    print("Saving output matrices")
    matrix = nib.Nifti1Image(out_matrix, header)
    outfile_name = output_name + '.nii'
    nib.save(matrix, outfile_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    shutil.move(os.path.join(os.getcwd(), outfile_name), os.path.join(output_dir, outfile_name))

if __name__ == "__main__":
    start = timer()

    parser = argparse.ArgumentParser()
    parser.add_argument("processed_epi", help="processed FMRI date (3d + time)")
    parser.add_argument("seed_mask", help="seed mask covering ROI (3d)")
    parser.add_argument("window_length", help="length of window in sliding-window analysis (trs)", type=int)
    parser.add_argument("overlap", help="overlap of windows (trs)", type=int)
    parser.add_argument("output_dir", help="output directory")
    args = parser.parse_args()
    
    data, seed, header = read_data(args.processed_epi, args.seed_mask)
    windows = calc_windows(args.window_length, args.overlap, data.shape[3])
    seed_tc = extract_seed_tc(seed, data)
    correlation = calculate_dynam_corr(windows, args.window_length, args.overlap, seed_tc, data)
#    correlation = calculate_static_corr(seed_tc, data)
    z_matrix = corr_to_z(correlation)
    output_name = os.path.basename(args.processed_epi)[:-4]
    save_output(correlation, header, output_name + "_DMNDynamCorr", args.output_dir)
    save_output(z_matrix, header, output_name + "_DMNDynamZ", args.output_dir)

    end = timer()
    print("Time Elapsed: " + str(end - start) + " s")