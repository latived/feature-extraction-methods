#!/usr/bin/env python

import math
import functools
from PIL import Image
import numpy # importar tudo msm é feio

# contém a imagem original: 1.tiff
img = Image.open("/home/lativ/Documents/UFAL/GioconDa/Dados/"
    "ImagesDataset/ColonDB/CVC-ColonDB/CVC-ColonDB/7.tiff"
    )

print("Caminho: {0}".format(img.filename))
print("Dimensões: {0}".format(img.size))

imgc = img.convert(mode="L") # RGB -> Cinza.
# eu deveria usar o .quantize em vez do convert?

PATCH_SIZE      = 50        # é o tamanho do patch 
GRAY_LEVELS     = 256       # imagem tem 24 níveis; requantizei para 8.
# TO DO: usar outros valores para displacement, como no artigo [1 GLCM pra cada]
# 0º  : [0, 1]
# 45º : [-1,1]
# 90º : [1, 0]
# 135º: [-1,-1]
displacement    = [1,1]     # padrão usado: significa um pixel para direita, um pixel para baixo (-45º)

# box=(left=0,  upper=12,   right=50,   lower=62) <- primeiro patch (superior esq)
# box=(left=450,upper=512,  right=500,  lower=562) <- último patch (inferior dir)

# TO DO: definir dimensões de patches de acordo com PATCH_SIZE
patches = [[None for _ in range(10)] for _ in range(11)]
for i in range(11):     # controla linhas
    for j in range(10): # controla colunas
            patches[i][j] = imgc.crop(
                    box=(
                        (j * PATCH_SIZE),                       # left
                        (12 + i * PATCH_SIZE ),                 # upper
                        (PATCH_SIZE + j * PATCH_SIZE),          # right (left + 50)
                        ((12 + PATCH_SIZE) + i * PATCH_SIZE)    # lower (upper + 50)
                    )
            )

# Agora já tenhos os 110 patches, mas alguns são pretos.
# Preciso descartar. Mas não eliminar-os-ei, por agora.

# ROWMAX e COLMAX são as bordas utilizadas para o cálculo da matriz GLCM
# Tá padrão do Avinash.
# Precisarei entender melhor (não é difícil) para quando for
# usar outras distâncias.
rowmax  =   PATCH_SIZE - displacement[0] if displacement[0] else PATCH_SIZE - 1
colmax  =   PATCH_SIZE - displacement[1] if displacement[1] else PATCH_SIZE - 1

# Matriz 11x10, onde cada célula é uma GLCM
patches_glcm = [[None for _ in range(10)] for _ in range(11)] 

# patch_i e patch_j localizam o patch que desejo na matriz acima.
for patch_i in range(11):       # patch posição i
    for patch_j in range(10):   # patch posição j
        # inicializa matriz GLCM
        glcm = [[0 for _ in range(GRAY_LEVELS)]
                for _ in range(GRAY_LEVELS)]
        # Image --> numpy.array
        image = numpy.asarray(patches[patch_i][patch_j])
        # Calculando GLCM para patches_glcm[patch_i][patch_j]
        for i in range(rowmax):
            for j in range(colmax):
                # pega valores (m,n) no padrão setado e incrementa glcm[m][n] e glcm[n][m]
                m, n = image[i][j], image[i + displacement[0]][j + displacement[1]]
                # simétrica
                glcm[m][n] += 1
                glcm[n][m] += 1
        # Uma glcm pronta, logo salvo
        patches_glcm[patch_i][patch_j] = glcm


# Para teste vou apenas selecionar uma glcm qualquer aqui
idx, jdx = 5, 4                 # indices do patch: linha, coluna
glcm = patches_glcm[idx][jdx]   # seleciono a glcm

# Formo o retângulo com (le,up) e (ri,lo)
le = jdx * 50       # left
up = 12 + idx * 50  # upper
ri = 50 + jdx * 50  # right
lo = 62 + idx * 50  # lower

print("Localização do patch:"
        " (left: {le}, upper: {up}, right: {ri}, lower: {lo})".format(
            le = le, up = up, ri = ri, lo = lo
            )
        )


# TO DO: salvar atributos em .csv? Vou usá-los com SVM, então...

# Calcular atributos
#   entropia, energia, constraste e homogeneidade

entropy = energy = contrast = homogeneity = None     # por que não 0 em vez de None?
# reduce(fun, seq[, initial]) -> value (isto é, reduz a matriz para a um valor)
normalizer = functools.reduce(lambda x, y: x + sum(y), glcm, 0) # GLCM: [0,1]
for m in range(GRAY_LEVELS):
    for n in range(GRAY_LEVELS):
        # Valores da GLCM como probabilidades
        prob = (1.0 * glcm[m][n]) / normalizer
        if (prob >= 0.0001) and (prob <= 0.999):
            log_prob = math.log(prob,2)  
        if prob < 0.0001:
            log_prob = 0
        if prob > 0.999:
            log_prob = 0
        # Por que a partir daqui os atributos são calculados apenas uma vez, no começo?
        # entropy is None, então calcula.
        # Aí passa pro próximo laço.
        # O mesmo acontece para os outros.
        if entropy is None:
            entropy = -1.0 * prob * log_prob
            continue   
        entropy += -1.0 * prob * log_prob
        if energy is None:
            energy = prob ** 2
            continue
        energy += prob ** 2
        if contrast is None:
            contrast = ((m - n) ** 2) * prob
            continue
        contrast += ((m - n) ** 2) * prob
        if homogeneity is None:
            homogeneity = prob / ( ( 1 + abs(m - n) ) * 1.0)
        homogeneity += prob / ( ( 1 + abs(m - n) ) * 1.0)
if abs(entropy) < 0.0000001: entropy = 0.0
print("\nTexture attributes: ")
print("     entropy: %f" % entropy)
print("     energyy: %f" % energy)
print("     contrast: %f" % contrast)
print("     homogeneity: %f" % homogeneity)
