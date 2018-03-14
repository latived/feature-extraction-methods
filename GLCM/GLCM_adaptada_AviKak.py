#!/usr/bin/env python

# Código feito por latived.
# Importante: objetivo desse código é aprendizado.

import math
import functools
import numpy as np
import csv 
import pymp

from PIL import Image

def particionar(img = None, ps=50): 
    #   recebe imagem e patch_size
    #   retorna lista de patches

    # TODO: checar possíveis valores de ps
    PATCH_SIZE  = ps
    lines = img.size[0] // ps
    cols = img.size[1] // ps

    # TODO: checar se é a melhor forma
    imgc = img.convert(mode="L") # RGB para Cinza.

    # box=(left=0,  upper=12,   right=50,   lower=62) <- primeiro patch (superior esq)
    # box=(left=450,upper=512,  right=500,  lower=562) <- último patch (inferior dir)

    # TODO: definir dimensões de patches de acordo com PATCH_SIZE
    # TODO: checar se é a melhor forma
    patches = [[None for _ in range(cols)] for _ in range(lines)]
    for i in range(11):     # controla linhas
        for j in range(10): # controla colunas
                # particiono a partir do 12o pixel para ficar melhor,
                # pois as imagems são 574x500
                # TODO: automatizar inicio da partição
                patches[i][j] = imgc.crop(
                        box=(
                            (j * PATCH_SIZE),                       # left
                            (12 + i * PATCH_SIZE ),                 # upper
                            (PATCH_SIZE + j * PATCH_SIZE),          # right (left + 50)
                            ((12 + PATCH_SIZE) + i * PATCH_SIZE)    # lower (upper + 50)
                        )   
                )   
    
    return patches # Eliminar patches majoritariamente pretos?

def calc_glcm(patch=None, disp=[0,1]):
    #       recebe patch
    #       retorna glcm

    displacement = disp # padrão usado
    
    # Image para ndarray
    image = np.asarray(patch)
    PATCH_SIZE = len(image)

    # ROWMAX e COLMAX são as bordas utilizadas para o cálculo da matriz GLCM
    rowmax  =   PATCH_SIZE - displacement[0] if displacement[0] else PATCH_SIZE - 1
    colmax  =   PATCH_SIZE - displacement[1] if displacement[1] else PATCH_SIZE - 1

    GRAY_LEVELS = 256 # total de níveis vai ser tam da GLCM

    # inicializa matriz GLCM
    shape = (GRAY_LEVELS, GRAY_LEVELS)
    glcm = np.zeros(shape=shape, dtype=np.int8)
    
    # Calculando GLCM para patch
    # TODO: verificar se há como melhorar 
    for i in range(rowmax):
        for j in range(colmax):
            # pega valores (m,n) no padrão setado e incrementa glcm[m][n] e glcm[n][m]
            m, n = image[i][j], image[i + displacement[0]][j + displacement[1]]
            # simétrica
            glcm[m][n] += 1
            glcm[n][m] += 1
    return glcm

def extrair_attrs(glcm = None):
    #       recebe glcm
    #       retorna lista de atributos

    # TODO: Verificar corretude das mudanças feitas.
    # 1. normalizer
    # 2. obteção das probabilidades
    # 3. calculo dos atributos

    GRAY_LEVELS = len(glcm)
   
    # Soma colunas e em seguida as linhas da matriz
    normalizer = np.add.reduce(np.add.reduce(glcm))

    # Normaliza
    probs = np.divide(glcm, normalizer) 
    probs_temp = np.copy(probs)
    # Fiz cópia acima para eliminar valores
    # fora do intervalo (0.0001,0.999)
    np.place(probs_temp, probs_temp < 0.0001, 1)
    np.place(probs_temp, probs_temp > 0.999, 1)
    # No caso prob = 1, temos log_prob = 0
    # Fiz isso de acordo com o original do Avinash.
    log_probs = np.log2(probs_temp)
    
    # TODO: explicar calculo dos atributos

    entropy = np.add.reduce(
                np.add.reduce(
                    np.multiply(
                        -probs, 
                        log_probs
                        )
                    )
                )

    energy = np.add.reduce(
            np.add.reduce(
                np.power(
                    probs, 
                    2
                    )
                )
            )

    # Matriz onde cada célula é LINHA - COLUNA
    msubn = [[line - col for col in range(GRAY_LEVELS)] for line in
            range(GRAY_LEVELS)]

    contrast = np.add.reduce(
                np.add.reduce(
                    np.multiply(
                        np.power(msubn, 2), 
                        probs
                        ) 
                    )
                )
    homogeneity = np.add.reduce(
                    np.add.reduce(
                        np.divide(
                            probs, 
                            1 + np.abs(msubn)
                            )
                        )
                    )
   
    # Usando numpy consegui reduzir 
    # a chamada de extrair_attrs de
    # ~0.5s para 0.015s
    # No geral, ficou de 55s para 5s, em média.

    """
    entropy = energy = contrast = homogeneity = 0
    normalizer = functools.reduce(lambda x, y: x + sum(y), glcm, 0)
    
    # Original:
    #    o gargalo tá aqui
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
    """

    if abs(entropy) < 0.0000001:
        entropy = 0.0
    
    attrs = [entropy, energy, contrast, homogeneity]

    return attrs

def salva_attrs(list_attrs_imgs, fname='attrs.csv', direcao):
    
    with open(fname, 'a', newline='') as fattrs:
        header = ['img',
                'patch',
                'direcao'
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
                
                writer.writerow({
                    'img': img_num,
                    'patch': patch_num,
                    'direcao': direcao,
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
 
    # TODO: avaliar este main

    # Atributos obtidos para cada patch de uma imagem
    list_attrs = [None for _ in range(110)]     # 110 porque já sei o total
    # Guardar um 'list_attrs' para cada imagem, para no fim escrever no arquivo
    list_attrs_imgs = [list_attrs for _ in range(300)]

    for i in range(1):
        print("Tratando imagem {}...".format(i+1))

        # Pega uma imagem e a salva em img
        img = Image.open(img_path + str(i+1) + ".tiff")
                
        ### CHAMADA DAS FUNÇÕES AQUI

        # Patches é uma lista com listas de partições da img original
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
