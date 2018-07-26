#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by lativ on 24/07/18 at 08:00
"""
import os
import sys
import numpy as np
import csv

from mahotas import features
from skimage import io
from skimage import color
from skimage import img_as_ubyte


def _glcm_6_other_attrs(patch):
    # Calculate INERTIA, CLUSTER SHADE, CLUSTER PROMINENCE
    # Inspired by mahotas
    _2d_deltas = [
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1)
    ]

    nr_dirs = len(_2d_deltas)
    fm1 = patch.max() + 1
    cmat = np.empty((fm1, fm1), np.int32)

    def all_cmatrices():
        for d in range(nr_dirs):
            features.texture.cooccurence(patch, d, cmat)
            yield cmat

    other_feats = []
    for cmat in all_cmatrices():
        feats = np.zeros(3, np.double)
        T = cmat.sum()
        maxv = len(cmat)
        k = np.arange(maxv)
        i, j = np.mgrid[:maxv, :maxv]
        i_j2 = (i - j) ** 2
        i_j2 = i_j2.ravel()

        p = cmat / float(T)
        pravel = p.ravel()
        px = p.sum(0)
        py = p.sum(1)

        ux = np.dot(px, k)
        uy = np.dot(py, k)

        i_j_ux_uy = (i + j - ux - uy)
        i_j_ux_uy3 = i_j_ux_uy ** 3
        i_j_ux_uy4 = i_j_ux_uy ** 4

        i_j_ux_uy3 = i_j_ux_uy3.ravel()
        i_j_ux_uy4 = i_j_ux_uy4.ravel()

        feats[0] = np.dot(i_j2, pravel)
        feats[1] = np.dot(i_j_ux_uy3, pravel)
        feats[2] = np.dot(i_j_ux_uy4, pravel)
        other_feats.append(feats)

    return np.array(other_feats).transpose()


def glcm_16_and_6(patch):

    """
    features_glcm_16 is a 16D vector with
        the features Energy, Local Homogeneity, Entropy and Correlation 
        for each one of the angles 0, 45, 90 and 135 .
    
    features_glcm_6 is a 6D vector with the mean values for the angles above and the
        features Energy, Local Homogeneity, Entropy, 
        Inertia, Cluster Shade and Cluster Prominence.
    """

    all_haralick = features.haralick(patch)  # All 13 (see method for about 14th feature)
    other_feats = _glcm_6_other_attrs(patch)  # Inertia, Cluster Shade and Prominence

    # Instead of a 4x4 matrix, here we go for a 16 dimensional vector.
    features_glcm_16 = np.concatenate([all_haralick[:, 0], all_haralick[:, 4], all_haralick[:, 8], all_haralick[:, 2]])

    features_glcm_6 = np.mean([all_haralick[:, 0],
                               all_haralick[:, 4],
                               all_haralick[:, 8],
                               other_feats[0],
                               other_feats[1],
                               other_feats[2]],
                              axis=1)

    return [features_glcm_16, features_glcm_6]


def lbp(patch):
    lbp_hist = features.lbp(patch, 1, 8)
    return lbp_hist


def lbp_oc(patch):
    pass


def read_image(folder, filename, gray=False):
    # Join the path to the image and the filename itself
    full_path_colon_img = os.path.join(folder, filename)
    # Read image
    # TODO: check if io.imread is the best method to read image
    colon_rgb = io.imread(full_path_colon_img)
    # Convert to gray
    # TODO: check for any information loss
    colon_gray = img_as_ubyte(color.rgb2gray(colon_rgb))

    if gray:
        return colon_gray

    return colon_rgb


def get_image_patches(img, patch_size):
    # TODO: find an automatic way to build this patches
    if patch_size == 50:
        patches = [
            [img[
             (line * 50):(50 + line * 50),
             (12 + col * 50):62 + col * 50
             ]
             for col in range(11)]
            for line in range(10)]
    elif patch_size == 70:
        patches = [
            [img[
             (5 + line * 70):(75 + line * 70),
             (7 + col * 70):(77 + col * 70)
             ]
             for col in range(8)]
            for line in range(7)]
    else:
        print("Invalid value for 'patch_size'. It needs to be 50 or 70.")
        sys.exit(0)

    # Instead of 10x11 matrices of 50x50, we go for 110x50x50.
    return np.concatenate(patches)


def patch_contains_polyp(patch, patch_size):
    white = np.count_nonzero(patch) / np.power(patch_size, patch_size)
    return white >= 0.50


def calculate_and_save_all_features(patches, patches_mask, patch_size, sequence):

    # TODO: add lbp_oc in the code

    methods = ['glcm_16', 'glcm_6', 'lbp']

    # FILES FOR EACH METHOD OF EXTRACTION
    files = list()
    for method in methods:
        # fn will be like 'seq01_ps50_glcm_16' for sequence=01 and patch_size=50
        fn = '_'.join(['seq' + str(sequence), 'ps' + str(patch_size), method])
        files.append(open('features_files/' + fn + '.csv', 'a', newline=''))

    # TODO: add new column 'polyp' to represent the presence of a polyp

    # FIELDNAMES (HEADERS) FOR EACH METHOD OF EXTRACTION
    fieldnames_glcm16 = ['energy_0', 'local_homogeneity_0', 'entropy_0', 'correlation_0',
                         'energy_45', 'local_homogeneity_45', 'entropy_45', 'correlation_45',
                         'energy_90', 'local_homogeneity_90', 'entropy_90', 'correlation_90',
                         'energy_135', 'local_homogeneity_135', 'entropy_135', 'correlation_135',
                         'polyp']
    fieldnames_glcm6 = ['energy_mean', 'local_homogeneity_mean', 'entropy_mean',
                        'inertia_mean', 'cluster_shade_mean', 'cluster_prominence',
                        'polyp']
    fieldnames_lbp = [str(id) for id in range(1, 37)] + ['polyp']
    # fieldnames_lbp_oc

    fieldnames = [fieldnames_glcm16, fieldnames_glcm6, fieldnames_lbp]

    # CSV DICT WRITER FOR EACH METHOD OF EXTRACTION
    writers = list()
    for (file, fn) in zip(files, fieldnames):
        w = csv.DictWriter(file, fn)  # Create DictWriter object
        if not csv.Sniffer().has_header(file):  # The header will be writen only in the first call
            w.writeheader()  # Writer header in the csv file
        writers.append(w)  # Add DictWriter object in the 'writers' list

    # For 50x50 matrices, we have 10x11 = 110 patches for each image
    # For 70x70 matrices, we have 7x8 = 56 patches for each image
    # Each patch has 16 + 6 + 36 + <lbp_oc_num_feats> = 58 features (without lbp_oc)
    # With 300 images, we have 300x100 + 300x56 = 49800 patches
    # Therefore, we have a ~50000x58 matrix of data to use in svm (without lbp_oc)
    for (patch, pmask) in zip(patches, patches_mask):
        feats_glcm16, feats_glcm6 = glcm_16_and_6(patch)
        feats_lbp = lbp(patch)
        # feats_lbp_oc = lbp_oc(patch)

        features = [feats_glcm16, feats_glcm6, feats_lbp]

        if patch_contains_polyp(pmask, patch_size):
            for f in features:
                f.append(1)
        else:
            for f in features:
                f.append(0)

        # Write the values of the features calculated in each respective file
        for (writer, fns, feats) in zip(writers, fieldnames, features):
            writer.writerow(dict(zip(fns, feats)))  # .writerow needs a dict with 'fieldname' : value for each cell

    # Close files here because it cost less than to open-and-close it all the time
    for file in files:
        file.close()


def svm_classifier():
    pass


def main():

    colondb_folder = "/home/lativ/GCD/Dados/ImagesDataset/ColonDB/CVC-ColonDB/CVC-ColonDB/"

    # Video sequence numbers of Bernal et al. (2012)
    sequences = (1, 2, 3, 5, 6, 7, 9, 10, 11, 14, 15)

    # TODO: Will I really need this 'filenames' list? Why not get the name on the run?

    # A dict with seq : [ filenames ], like: '1' : ['1.tiff', ..., '39.tiff']
    filenames = dict()
    filenames_mask = dict()  # filenames for the polyp masks
    filenames[sequences[0]] = [str(id) + '.tiff' for id in range(1, 39)]
    filenames_mask[sequences[0]] = ['p' + str(id) + '.tiff' for id in range(1, 39)]

    # TODO: Modify RHS below
    """
    filenames[sequences[1]] = list(range(39, 61))
    filenames[sequences[2]] = list(range(61, 77))
    filenames[sequences[3]] = list(range(77, 98))
    filenames[sequences[4]] = list(range(98, 149))
    filenames[sequences[5]] = list(range(149, 156))
    filenames[sequences[6]] = list(range(156, 204))
    filenames[sequences[7]] = list(range(204, 209))
    filenames[sequences[8]] = list(range(209, 264))
    filenames[sequences[9]] = list(range(264, 274))
    filenames[sequences[10]] = list(range(274, 301))
    """

    # For each sequence,
    for seq in sequences:
        # For each filename (like '1.tiff', or '39.tiff')

        # Just to don't raise KeyError
        if seq not in filenames.keys():
            filenames[seq] = list()
            filenames_mask[seq] = list()

        for (filename, filename_mask) in zip(filenames[seq], filenames_mask[seq]):
            colon_gray = read_image(colondb_folder, filename, gray=True)
            colon_mask = read_image(colondb_folder, filename_mask)  # The polyp mask is a binary image

            # Get the patches lists
            patches_50 = get_image_patches(colon_gray, 50)
            patches_70 = get_image_patches(colon_gray, 70)

            # Get the patches lists for the polyp mask
            patches_50_mask = get_image_patches(colon_mask, 50)
            patches_70_mask = get_image_patches(colon_mask, 70)

            # Do the freaking thing
            calculate_and_save_all_features(patches_50, patches_50_mask, 50, seq)
            calculate_and_save_all_features(patches_70, patches_70_mask, 70, seq)


if __name__ == '__main__':
    """
    Test before run it.
    """
    main()
