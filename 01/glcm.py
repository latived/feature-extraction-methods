#!/usr/bin/env python

# Código feito por latived, baseado no tutorial "Measuring Texture and Color"
# Replicação do artigo: Texture-based polyp detection in colonoscopy

# Funções:
#   calc_glcm(patch, disp)  -> retorna glcm para o patch usando um displacement
#   calc_attrs(glcm)        -> retorn atributos extraídos da glcm 

import numpy as np

from PIL import Image # TODO: analisar uso do scikit-image no lugar.

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

def calc_attrs(glcm = None):
    #       recebe glcm
    #       retorna lista de atributos

    # TODO: inserir novos atributos de acordo com artigo original (ver linha 5)

    # TODO: Verificar corretude das mudanças feitas.
    # 1. normalizer
    # 2. obtenção das probabilidades
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
   
    if abs(entropy) < 0.0000001:
        entropy = 0.0
    
    attrs = [entropy, energy, contrast, homogeneity]

    return attrs
