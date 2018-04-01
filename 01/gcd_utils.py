#!/usr/bin/env python

import csv
import numpy as np

from PIL import Image # TODO: analisar uso do scikit-image no lugar.

# Nesse módulo temos: 
#   calc_patches(img, ps) -> retorna matriz de patches para img de acordo com ps
#   salva_attrs(flist_attrs_imgs, direcao, name) -> salva em arquivo csv os
#       atributos calculados para direcao na contidos flist_attrs_imgs

# TODO: refatorar.
def calc_patches(img = None, ps=50):
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
                patches[i][j] = np.asarray(imgc.crop(
                        box=(
                            (j * PATCH_SIZE),                       # left
                            (12 + i * PATCH_SIZE ),                 # upper
                            (PATCH_SIZE + j * PATCH_SIZE),          # right (left + 50)
                            ((12 + PATCH_SIZE) + i * PATCH_SIZE)    # lower (upper + 50)
                        )
                ))

    return patches # Eliminar patches majoritariamente pretos?

# TODO: passar header/lista de atributos como parâmetro
# TODO: passar método também? para essa função ficar mais genérica?
# TODO: é realmente um utilitario?
def salva_attrs(list_attrs_imgs, direcao, fname='attrs.csv'):
    
    with open(fname, 'a', newline='') as fattrs:
        header = ['img',
                'patch',
                'direcao',
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

    print("Atributos salvos com sucesso no arquivo {}".format(fname))
