#!/usr/bin/env python

# Código feito por latived, versão noob++. Talvez tenha pouca performance.
# Importante: objetivo desse código é aprendizado.

import math
import functools
import numpy as np
import csv 
import pymp

from PIL import Image

#   recebe imagem e patch_size
#   retorna lista de patches
def particionar(img = None, ps=50):
    PATCH_SIZE  = ps    # é o tamanho do patch 

    # eu deveria usar o .quantize em vez do convert?
    imgc = img.convert(mode="L") # RGB -> Cinza.

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
    
    return patches
    # 110 patches se ps=50, mas alguns são pretos.

#       recebe patch
#       retorna glcm
def calc_glcm(patch=None, disp=[0,1]):

    displacement = disp # padrão usado
    
    # Image --> numpy.array
    image = np.asarray(patch)
    PATCH_SIZE = len(image)

    # ROWMAX e COLMAX são as bordas utilizadas para o cálculo da matriz GLCM
    rowmax  =   PATCH_SIZE - displacement[0] if displacement[0] else PATCH_SIZE - 1
    colmax  =   PATCH_SIZE - displacement[1] if displacement[1] else PATCH_SIZE - 1

    GRAY_LEVELS = 256 # total de níveis vai ser tam da GLCM

    # inicializa matriz GLCM
    shape = (GRAY_LEVELS, GRAY_LEVELS)
    glcm = np.zeros(shape=shape, dtype=np.int8)
    #"""
    # Calculando GLCM para patch
    for i in range(rowmax):
        for j in range(colmax):
            # pega valores (m,n) no padrão setado e incrementa glcm[m][n] e glcm[n][m]
            m, n = image[i][j], image[i + displacement[0]][j + displacement[1]]
            # simétrica
            glcm[m][n] += 1
            glcm[n][m] += 1
    return glcm

#       recebe glcm
#       retorna lista de atributos
def extrair_attrs(glcm = None):
    GRAY_LEVELS = len(glcm)

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
    
    attrs = [entropy, energy, contrast, homogeneity]

    return attrs

def salva_attrs(list_attrs_imgs, file_name='attrs.csv'):
    fname = file_name
    with open(fname, 'a', newline='') as fattrs:
        header = ['img',
                'patch',
                'entropia',
                'energia',
                'contraste',
                'homogeneidade']

        writer = csv.DictWriter(fattrs, fieldnames=header)
        writer.writeheader() # e se o arquivo já existir?

        img_num = 0
        print("Salvando atributos da imagem {}".format(img_num+1))
        # lattrs: 110 lists do tipo [at1,at2,at3,at4]
        for lattrs in list_attrs_imgs:  
            img_num += 1
            patch_num = 0
            for attrs in lattrs: # attr é [at1,at2,at3,at4]
                patch_num += 1
                # Escreve num arquivo csv
                writer.writerow({
                    'img': img_num,
                    'patch': patch_num,
                    'entropia': attrs[0],
                    'energia': attrs[1],
                    'contraste': attrs[2],
                    'homogeneidade': attrs[3]
                    })
    print("Atributos salvos com sucesso no arquivo {}".format(file_name))

def main():
     # Talvez modifique para recebê-la como entrada
    img_path = ("/home/lativ/Documents/UFAL/GioconDa/Dados/ImagesDataset/"
            "ColonDB/CVC-ColonDB/CVC-ColonDB/")
 
    # Atributos obtidos para cada patch de uma imagem
    list_attrs = [None for _ in range(110)]     # 110 porque já sei o total
    # Guardar um 'list_attrs' para cada imagem, para no fim escrever no arquivo
    list_attrs_imgs = [list_attrs for _ in range(300)]

    # Cada imagem toma ~1min, logo tudo dá ~5h.
    ## Paralelizando glcm(): ~55 segundos por imagem. 
    ##      Nada melhorado. Algo eu fiz errado. 
    ##      Tá 2x pior.
    ## Preciso estudar OpenMP.
    ## Eu poso tentar paralelizar aqui usando listas, 
    ## que diz ser um pouco mais lento, mas deve ser o bastante. 
    for i in range(1):
        print("Tratando imagem {}...".format(i+1))

        # pega 1 imagem e salva em img
        img = Image.open(img_path + str(i+1) + ".tiff")
                
        ### CHAMADA DAS FUNÇÕES AQUI

        # patches é uma lista com listas de partições da img original
        patches = particionar(img) # matriz de patches 11x10
        line_num = 0
        for line in patches:
            patch_num = 0
            for patch in line:
                # Calculando GLCM de um patch 
                glcm = calc_glcm(patch) # lembre que displacement = [0,1] apenas
                # Extração dos 4 atributos
                # imgs_list_attrs[i] index os attrs da img i
                # e imgs...[line_num * 10 + patch_num] salva os attrs do patch
                list_attrs_imgs[i][line_num * 10 + patch_num] = extrair_attrs(glcm)
                patch_num += 1
            line_num += 1

    salva_attrs(list_attrs_imgs)
   

if __name__ == "__main__":
    main()
