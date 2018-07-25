#!/usr/bin/env python3
import os
from skimage import io
from skimage import color
from skimage import img_as_ubyte
from mahotas import features

import numpy as np

# TODO: use all images in path
img_path = "/home/lativ/Documents/UFAL/GioconDa/Dados/ImagesDataset/ColonDB/CVC-ColonDB/CVC-ColonDB/"
filename = os.path.join(img_path, "1.tiff")
# TODO: verify the need to use mahotas.imread?
# TODO: verify the need to use as_grey parameter?
colon = io.imread(filename)

# colon_gray is in float64, therefore I convert to ubyte (integer of 8 bits)

colon_gray = img_as_ubyte(color.rgb2gray(colon))  # TODO: is there a better way?
# TODO: verify the need to use mahotas.strech(img)

# TODO: cut each image in patches of 50x50 and 70x70

# For 50x50 we have 10x11 patches.
# For 70x70 we have 7x8 patches.

# colon_gray.shape -> (500, 574)
# colon_gray[0:50, 12:62] ... [450:500, 512:562]
# colon_gray[5:75, 7:77] .. [424:495, 497:567]

# TODO: find an automatic way to build this patches
patches_50 = [
    [colon_gray[
     (line * 50):(50 + line * 50),
     (12 + col * 50):62 + col * 50
     ]
     for col in range(11)]
    for line in range(10)]

patches_70 = [
    [colon_gray[
     (5 + line * 70):(75 + line * 70),
     (7 + col * 70):(77 + col * 70)
     ]
     for col in range(8)]
    for line in range(7)]

# TODO: Haralick features
# GLCM 16:
#   d = 1, alpha em (0, 45, 90, 135), 
#       energy, 
#       homogeneity,
#       entropy e 
#       correlation
#   result: 16-dimension vector, where [at1,at2,at3,at4] for each d
# GLCM 6:
#   d = 1, mean of the 4 directions, and with features
#       energy, 
#       homogeneity,
#       entropy, 
#       inertia,
#       cluster shade,
#       cluster prominence
#   result: 6-dimension vector like [mean_at1... mean_at6]

# TODO: use patches
actual_patch = patches_50[5][5]
# actual_patch = patches_70[5][5]
hfeats = features.haralick(actual_patch)
# TODO: is there any better way here?
feats_glcm_16 = list()
# http://www.ucalgary.ca/mhallbey/asm
# Should I get sqrt of hfetas[:,0]? Yes... actually won't.
# In other places we have -> Energy = ASM, hm...
feats_glcm_16.append(hfeats[:, 0])  # Energy = Homogeneity = Angular Second Moment
feats_glcm_16.append(hfeats[:, 4])  # Local Homogeneity (or Inverse Difference Moment)
feats_glcm_16.append(hfeats[:, 8])  # Entropy
feats_glcm_16.append(hfeats[:, 2])  # Correlation

feats_glcm_6 = list(np.mean([hfeats[:, 0], hfeats[:, 4], hfeats[:, 8]], axis=1))
# Convert to list is good as it is above?


# Inertia resembles the contrast feature
# Cluster shade and prominence aren't calculated in Haralick's '73 paper
# See Conners et al 1984 instead.

# Now I will calculate the three features below.

# inertia    = \sum_{i=0}^{G-1} \sum_{j=0}^{G-1} {i - j}^2 * P(i,j)
# shade      = \sum_{i=0}^{G-1} \sum_{j=0}^{G-1} {i + j - ux - uy}^3 * P(i,j) 
# prominence = \sum_{i=0}^{G-1} \sum_{j=0}^{G-1} {i + j - ux - uy}^4 * P(i,j) 

_2d_deltas = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1)]
nr_dirs = len(_2d_deltas)
fm1 = actual_patch.max() + 1
cmat = np.empty((fm1, fm1), np.int32)


def all_cmatrices():
    for d in range(nr_dirs):
        features.texture.cooccurence(actual_patch, d, cmat, symmetric=True, distance=1)
        yield cmat


rfeatures = []
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
    rfeatures.append(feats)

# TODO: check again values (high enough to put me in doubt about it)
# TODO: how to check it?

# I already had done the for loop above, without vectorization,
# and the results were all the same but with a differences in the order of e-14.

rfeatures = np.array(rfeatures)
rfeatures_mean = rfeatures.mean(axis=0)

feats_glcm_6.append(rfeatures_mean[0])  # inertia
feats_glcm_6.append(rfeatures_mean[1])  # cluster shade
feats_glcm_6.append(rfeatures_mean[2])  # cluster prominence

# TODO: LBP feature

# Não entendi porque points é flexível
#   Consquência da fórmula.
# Não entendi porque lbp_hist é 36
#   É porque há apenas 36 diferentes códigos de 8 bits que sejam invariantes à rotação.
# O paper usa 64 atributos, logo essa reprodução já não vai ser exata.
lbp_hist = features.lbp(actual_patch, 1, 8)


# TODO: OC-LBP feature
