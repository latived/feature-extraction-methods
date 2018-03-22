#!/usr/bin/env python

# by latived [210318]

import gcd_utils
import glcm
import lbp

# TODO: não acho que deveria particionar aqui também
#       ou seria não particionar
from PIL import Image

def main():
     # Talvez modifique para recebê-la como entrada
    img_path = ("/home/lativ/Documents/UFAL/GioconDa/Dados/ImagesDataset/"
            "ColonDB/CVC-ColonDB/CVC-ColonDB/")
 
    # TODO: avaliar este main

    # Atributos obtidos para cada patch de uma imagem
    list_attrs = [None for _ in range(110)]     # 110 porque já sei o total
    # Guardar um 'list_attrs' para cada imagem, para no fim escrever no arquivo
    list_attrs_imgs = [list_attrs for _ in range(300)]
    
    # displacement/direcao
    disp = [0,1]

    for i in range(1):
        print("Tratando imagem {}...".format(i+1))

        # TODO: em vez de passar Image object, seria melhor passar seu caminho,
        # não?

        # Pega uma imagem e a salva em img
        img = Image.open(img_path + str(i+1) + ".tiff")
                
        ### CHAMADA DAS FUNÇÕES AQUI

        # Patches é uma lista com listas de partições da img original
        patches = gcd_utils.calc_patches(img) # matriz de patches 11x10
        line_num = 0
        for line in patches:
            patch_num = 0
            for patch in line:
                # Calculando GLCM de um patch 
                glcm_matrix = glcm.calc_glcm(patch, disp)
                # Extração dos 4 atributos
                # imgs_list_attrs[i] index os attrs da img i
                # e imgs...[line_num * 10 + patch_num] salva os attrs do patch
                list_attrs_imgs[i][line_num * 10 + patch_num] = \
                    glcm.calc_attrs(glcm_matrix)
                patch_num += 1
            line_num += 1

    gcd_utils.salva_attrs(list_attrs_imgs, disp)

if __name__ == "__main__":
    main()
