#!/usr/bin/env python

# Código feito por latived, versão noob. Talvez tenha pouca performance.
# Importante: objetivo desse código é aprendizado.

import math
import functools
from PIL import Image
import numpy 
import csv 

print("Iniciando extração de atributos: GLCM")

# A idéia do código é:
# 1. receber uma imagem de videocolonoscopia            [feito]
# 2. converter para escala de cinza                     [feito, mas dúvidas]
# 3. particionar em patches de (atualmente) 50x50       [feito]
# 4. calcular matriz de co-ocorrência por patch         [1 de 4 direções][falta]
#        extrair os atributos                           [done]
#        e salvar os atributos desejados 
#           em um arquivo .csv                          [done]

# Item 2: resultado ok, mas dúvida se é o melhor modo
# Item 4: no artigo é obtida 1 matriz para cada distância (displacement)

### Etapa 1: contém a imagem original 
# Talvez modifique para recebê-la como entrada
img = Image.open("/home/lativ/Documents/"
        "UFAL/GioconDa/Dados/"
        "ImagesDataset/ColonDB/"
        "CVC-ColonDB/CVC-ColonDB/7.tiff"
        )

print("Caminho da imagem: {0}".format(img.filename))
print("Imagem carregada!")

### Etapa 2: conversão para escala de cinza
imgc = img.convert(mode="L") # RGB -> Cinza.
# eu deveria usar o .quantize em vez do convert?

print("Conversão para escala de cinza feita!")

### Etapa 3: particionando imagem original em patches
print("Preparando particionamento...")

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
            
            
print("Particinamento completo!")
print("=> Tamanho do patch usado: "
        "{patch_size}x{patch_size}".format(patch_size=PATCH_SIZE))

total_patches = len(patches) * len(patches[0]) # 11 * 10 para PATCH_SIZE=50
print("=> Total de patches: {total}".format(total=total_patches))

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

### Etapa 4: calcular matriz de co-ocorrência, extrair atributos e salvá-los
print("Calculando GLCM para cada patch.")
with open('attrs.csv', 'w', newline='') as attrs:
    header = ['patch',
            'entropia',
            'energia',
            'contraste',
            'homogeneidade']
    writer = csv.DictWriter(attrs, fieldnames=header)
    writer.writeheader()

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
            # já posso calcular os atributos aqui?
            # mudando de None para 0
            # não consigo ver algum problema por isso
            entropy = energy = contrast = homogeneity = 0
            normalizer = functools.reduce(lambda x, y: x + sum(y), glcm, 0)
            for m in range(GRAY_LEVELS):
                for n in range(GRAY_LEVELS):
                    prob = (1.0 * glcm[m][n]) / normalizer
                    if (prob >= 0.0001) and (prob <= 0.999):
                        log_prob = math.log(prob, 2)
                    if prob < 0.0001:
                        log_prob = 0
                    if prob > 0.999:
                        log_prob = 0
                    entropy += -1.0 * prob * log_prob
                    energy += prob ** 2
                    contrast += ((m - n) ** 2) * prob
                    homogeneity += prob / ((1 + abs(m - n)) * 1.0)
            if abs(entropy) < 0.0000001:
                entropy = 0.0
            # salvar os atributos em um arquivo
            writer.writerow({'patch': 10 * (patch_i) + patch_j + 1,
                'entropia': entropy,
                'energia': energy,
                'contraste': contrast,
                'homogeneidade': homogeneity
                })


print("GLCMs construídas, atributos extraídos e salvos com sucesso.")

