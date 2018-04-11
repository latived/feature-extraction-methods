#!/usr/bin/env python3

# Documentar as coisas aqui.
# Por que estou usando mahotas?

import os
from skimage import io
from skimage import color
from skimage import img_as_ubyte
from mahotas import features

# TODO: automatizar para todas imagens que tenho
img_path = "/home/lativ/Documents/UFAL/GioconDa/Dados/ImagesDataset/ColonDB/CVC-ColonDB/CVC-ColonDB/"
filename = os.path.join(img_path, "1.tiff")
colon = io.imread(filename) # TODO: usar mahotas.imread?

# INFO: colon_gray está em float64, logo converto pra ubyte (inteiro 8 bits)
# INFO: Precisão pode ser perdida.
colon_gray = img_as_ubyte(color.rgb2gray(colon)) # TODO: há outra melhor forma?
# TODO: ver mahotas.strech(img)

# TODO: cortar cada imagem em patches 50x50 e 70x70
# algo assim: colon_gray[50:100, 50:100]

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

# TODO: Calcular hfeats para todas distâncias
hfeats = features.haralick(colon_gray)
# TODO: Melhorar essa parte
feats_glcm_16 = []
feats_glcm_16.append(hfeats[:,0]) # Energy
feats_glcm_16.append(hfeats[:,4]) # Homogeneity
feats_glcm_16.append(hfeats[:,7]) # Entropy
feats_glcm_16.append(hfeats[:,2]) # Correlation

# GLCM 6
# use parâmetro 'return_mean'
# ...
# inertia?
# cluster_shade?
# cluster_prominence?

# TODO: computar GLCM 16 e GLCM 6

# TODO: LBP feature

# Não entendi porque points é flexível
# Não entendi porque lbp_hist é 36
lbp_hist = features.lbp(colon_gray, 1, 8) 


# TODO: OC-LBP feature
