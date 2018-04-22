#!/usr/bin/env python3

# Documentar as coisas aqui.
# Por que estou usando mahotas?

import os
from skimage import io
from skimage import color
from skimage import util
from skimage import img_as_ubyte
from mahotas import features

import numpy as np

# TODO: automatizar para todas imagens que tenho
img_path = "/home/lativ/Documents/UFAL/GioconDa/Dados/ImagesDataset/ColonDB/CVC-ColonDB/CVC-ColonDB/"
filename = os.path.join(img_path, "1.tiff")
# TODO: usar mahotas.imread?
# TODO: usar as_grey parameter?
colon = io.imread(filename) 

# INFO: colon_gray está em float64, logo converto pra ubyte (inteiro 8 bits)

colon_gray = img_as_ubyte(color.rgb2gray(colon)) # TODO: há outra melhor forma?
# TODO: ver mahotas.strech(img)

# TODO: cortar cada imagem em patches 50x50 e 70x70

# para 50x50 temos 10x11 patches
# para 70x70 temos 7x8 patches

# colon_gray.shape -> (500, 574)
# colon_gray[0:50, 12:62] ... [450:500, 512:562]
# colon_gray[5:75, 7:77] .. [424:495, 497:567]

# TODO: encontrar uma maneira automática de construir esses patches
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
#
# GLCM 16: 
#   d = 1, alpha em (0, 45, 90, 135), 
#       energy, 
#       homogeneity,
#       entropy e 
#       correlation
#   resultado: vetor 16 dimensões, onde [at1,at2,at3,at4] para cada
# GLCM 6:
#   d = 1, média das 4 direções, e extraindo os atributos
#       energy, 
#       homogeneity,
#       entropy, 
#       inertia,
#       cluster shade,
#       cluster prominence
#   resultado: veotr 6 dimensões, onde [at1...at6]

# TODO: usar patches em vez da imagem inteira
# TODO: Calcular hfeats para todas distâncias
hfeats = features.haralick(colon_gray)
# TODO: Melhorar essa parte
feats_glcm_16 = []
# http://www.ucalgary.ca/mhallbey/asm
# Should I get sqrt of hfetas[:,0]? Yes... actually won't.
# Em outros locais Energy = ASM, hm...
feats_glcm_16.append(hfeats[:,0]) # Energy = Homogeneity = Angular Second Moment
feats_glcm_16.append(hfeats[:,4]) # Local Homogeneity (or Inverse Difference Moment)
feats_glcm_16.append(hfeats[:,8]) # Entropy
feats_glcm_16.append(hfeats[:,2]) # Correlation

# TODO: bom? Não poderia eu pegar a média de hfeats? 
hfeats_mean = features.haralick(colon_gray, return_mean=True)
feats_glcm_6 = []

feats_glcm_6.append(hfeats_mean[:,0]) # Energy
feats_glcm_6.append(hfeats_mean[:,4]) # Local homogeneity
feats_glcm_6.append(hfeats_mean[:,8]) # Entropy

# inertia assemelha-se a contraste, enquanto cluster shade e prominence
# não são calculadas pois são novas (pós haralick) (ver Conners et al 1984)

# inertia    = \sum_{i=0}^{G-1} \sum_{j=0}^{G-1} {i - j}^2 * P(i,j)
# shade      = \sum_{i=0}^{G-1} \sum_{j=0}^{G-1} {i + j - ux - uy}^3 * P(i,j) 
# prominence = \sum_{i=0}^{G-1} \sum_{j=0}^{G-1} {i + j - ux - uy}^4 * P(i,j) 
_2d_deltas = [
        (0,1),
        (1,1),
        (1,0),
        (1,-1)]
nr_dirs = len(_2d_deltas)
fm1 = colon_gray.max() + 1 # por que + 1?
cmat = np.empty((fm1, fm1), np.int32)
def all_cmatrices():
    for dir in range(nr_dirs):
        glcm = features.texture.cooccurence(colon_gray, dir, cmat,
                symmetric=True, distance=distance)
        yield cmat

rfeatures = []
for cmat in all_cmatrices():
    feats   = np.zeros(3)
    T       = cmat.sum()
    maxv    = len(cmat)
    k       = np.arange(2*maxv)
    i,j     = np.mgrid(:maxv,:maxv)
    i_j2 = (i - j)**2
    i_j2 = i_j2.ravel()
    
    p       = cmat / float(T)
    pravel  = p.ravel()
    px      = p.sum(0)
    py      = p.sum(1)

    ux = np.dot(px, k)
    uy = np.dot(py, k)
    
    i_j_ux_uy = (i + j - ux - uy)
    i_j_ux_uy3 = i_j_ux_uy**3
    i_j_ux_uy4 = i_j_ux_uy**4
    
    feats[0] = np.dot(i_j2, pravel)
    feats[1] = np.dot(i_j_ux_uy3, pravel)
    feats[2] = np.dot(i_j_ux_uy4, pravel)
    rfeatures.append(feats)

rfeatures_mean = rfeatures.mean(axis=0)

feats_glcm_6.append(rfeatures_mean[:,0]) # inertia
feats_glcm_6.append(rfeatures_mean[:,1]) # cluster shade
feats_glcm_6.append(rfeatures_mean[:,2]) # cluster prominence

# TODO: LBP feature

# Não entendi porque points é flexível
# Não entendi porque lbp_hist é 36
lbp_hist = features.lbp(colon_gray, 1, 8) 


# TODO: OC-LBP feature
