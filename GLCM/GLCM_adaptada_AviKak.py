#!/usr/bin/env python

# para que isso acima mesmo?

# Qual a importância de criar uma matriz simétrica? no que isso ajuda? é melhor?
## : os níveis m e n ocorrem juntos, é isso que importa apenas.

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

# requantizar (nome correto?) 24 -> 8 bits 
# (acho que é basicamente cor -> cinza)
imgc = img.convert(mode="L") # pronto. imgc é 7.tiff em b&p
# eu deveria usar o .quantize ou .convert msm?

# PATCH_SIZE = 50
# no nosso caso temos sempre: (574, 500) 
# e pelo que vi, dá pra supor que as primeiras e últimas linhas são todas de valor 0
# PRA VERIFICAR: posso remover 12 linhas em cima e 12 linhas embaixo?
# supondo que sim...
# então uma partição de 50x50 nos dá: 11 x 10 patches = 110 patches
# como temos 300 imagens, isso totaliza 33000 patches
# um dos problemas em ter muitos patches assim é que poucos são realmente de casos anormais (com pólipos)

# como guardar os patches? uma lista 11x10 é a solução mais óbvia
# box=(left=0,upper=12,right=50,lower=62) <- primeiro patch (superior esq)
# (450,512,500,562) <- último patch (inferior dir)
# tenho a imagem no objeto Image 'imgc', usarei .crop
patches = [[None for _ in range(10)] for _ in range(11)] # 10 col, 11 linha ; deveria inicializar melhor?
for i in range(11):     # controla linhas
    for j in range(10): # controla colunas
            patches[i][j] = imgc.crop(
                    box=(
                        (j * 50),       # left
                        (12 + i * 50),  # upper
                        (50 + j * 50),  # right (left + 50)
                        (62 + i * 50)   # lower (upper + 50)
                    )
            )

# agora já tenhos os 110 patches
# alguns são pretos, então preciso descartar. 
# como? mais 50% dos pixels serem pretos?
# ou elimino a primeira e última linhas e colunas? 
# melhor verificar por valor = 0; há imagens que contém muitas colunas pretas
# há também imagens com valores baixos mas não pretos
# serão afetadas?
# decisão: não eliminar patches agora.

# requantizei la no começo

# IMAGE_SIZE      = 8         # tamanho da matriz, mas acho que usarei valor diferente
# GRAY_LEVELS     = 6         # número níveis de cinza da imagem, geralmente 256 para uma de 8 bits
IMAGE_SIZE      = 50        # é o tamanho do patch msm
GRAY_LEVELS     = 256       # imagem tem 24 níveis; requantizei para 8 bits
# talvez eu faça mais... depois...
displacement    = [1,1]     # padrão usado: significa um pixel para direita, um pixel para baixo (-45º)

# ROWMAX e COLMAX são as bordas utilizadas para o cálculo da matriz GLCM
# tá padrão do Avi; precisarei entender melhor (não é difícil) para quando for
# usar outras distâncias
rowmax  =   IMAGE_SIZE - displacement[0] if displacement[0] else IMAGE_SIZE - 1
colmax  =   IMAGE_SIZE - displacement[1] if displacement[1] else IMAGE_SIZE - 1

# matriz 11x10 com cade célula sendo uma glcm
patches_glcm = [[None for _ in range(10)] for _ in range(11)] # 10 col, 11 linha ; deveria inicializar melhor?
# depois eu melhoro essa parte (deixar mais clara e mais eficiente)
# len(patches) dps coloco aqui no lugar de 10 e 11
for patch_i in range(11):       # patch posição i
    for patch_j in range(10):   # patch posição j
        # inicializa matriz GLCM
        glcm = [[0 for _ in range(GRAY_LEVELS)]
                for _ in range(GRAY_LEVELS)]
        # converter de Image para um numpy.array
        image = numpy.asarray(patches[patch_i][patch_j])
        # pronto: image é um dos 110 patches
        # quanto tempo leva as instruções abaixo? O(n^2), né. n = 256 (tam. glcm)
        for i in range(rowmax):
            for j in range(colmax):
                # pega valores (m,n) no padrão setado e incrementa glcm[m][n] e glcm[n][m]
                m, n = image[i][j], image[i + displacement[0]][j + displacement[1]]
                # simétrica
                glcm[m][n] += 1
                glcm[n][m] += 1
        # terminei uma glcm, então a salvo
        patches_glcm[patch_i][patch_j] = glcm       # depois refatoro o código 


# para teste vou apenas selecionar uma glcm qualquer aqui
idx, jdx = 7, 5
glcm = patches_glcm[idx][jdx]
le = jdx * 50
up = 12 + idx * 50
ri = 50 + jdx * 50
lo = 62 + idx * 50
print("Localização do patch:"
        " (left: {le}, upper: {up}, right: {ri}, lower: {lo})".format(le=le,up=up,ri=ri,lo=lo))

# calcular atributos
#   entropia, energia, constraste e homogeneidade

entropy = energy = contrast = homogeneity= None     # por que não 0 em vez de None?
# reduce(fun, seq[, initial]) -> value
# parece que reduz linhas e colunas a um numero só (soma tudo)
normalizer = functools.reduce(lambda x, y: x + sum(y), glcm, 0) # vai ser usado pra normalizar os valores em (0,1)
for m in range(len(glcm[0])):
    for n in range(len(glcm[0])):
        prob = (1.0 * glcm[m][n]) / normalizer      # não entendi o 1.0 ali... 
        if (prob >= 0.0001) and (prob <= 0.999):
            log_prob = math.log(prob,2)             # qual seria o sentido de log_prob? tipo, que representa pra imagem?
        if prob < 0.0001:
            log_prob = 0
        if prob > 0.999:
            log_prob = 0
        # por que a partir daqui os atributos são calculados apenas uma vez, no começo?
        # entropy is None, então calcula, aí passa pro próximo laço
        # o mesmo acontece para os outros
        # assim os 4 primeiros ciclos cada um serve pra inicializar um atributo
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
