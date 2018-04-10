#!/usr/bin/env python3

# Documentar as coisas aqui
# Por que estou usando mahotas?

import os
from skimage import io
from skimage import color
from skimage import img_as_ubyte
from mahotas import features


filename = os.path.join(img_path, "1.tiff")
colon = io.imread(filename)

# TODO: cortar imagem em patches 50^2 e 70^2

# Haralick

# hfeats é uma matriz de direções vs atributos
hfeats = features.haralick(colon_gray)
# TODO: computar GLCM 16 e GLCM 6

# LBP

# colon_gray está em float64,
# logo boto pra ubyte (inteiro 8 bits)
# Precisão pode ser perdida.
colon_gray =  img_as_ubyte(color.rgb2gray(colon))
# Não entendi porque points é flexível
# Não entendi porque lbp_hist é 36
lbp_hist = features.lbp(colon_gray, 1, 8) 


# TODO: OC-LBP
