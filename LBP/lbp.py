#!/usr/bin/env python

import random
import math
import BitVector

import numpy as np

# Vejamos, no artigo original temos (TBPDIC):
#   LBP is extracted from a 3x3 nb
#   The nb is binarised ...
#   The values (runs?) are then multiplied by certain weights and summed (1)
#   Leading to one single value for each nb.
#   .. an amount of 64 histogram bins was used
#   leading to a 64-dimensional feature vector.
# Num patch 50x50, usando uma matriz 3x3 temos para cada matriz:
#   LBP_P,R (x_c, y_c) = sum p = 0 até P - 1 de s(g_p - g_c)2^p (2)
#   LBP_P,R resulta em 2^P valores distintos
#   Nessa matriz temos P = 8, logo 2^8 = 256 valores   
#   Porém... detalhe: no artigo o histograma tem 64 bins, não 256. 
##
#   Essa formula acima peguei de: 2003-TheLocalBinaryPatternApproach

IMAGE_SIZE = 50                                                                    #(A6)
GRAY_LEVELS = 256                                                                   #(A7)
R = 1                  # the parameter R is radius of the circular pattern        #(A8)
P = 8                  # the number of points to sample on the circle             #(A9)

lbp = [[0 for _ in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)]                 #(C4)
rowmax,colmax = IMAGE_SIZE-R,IMAGE_SIZE-R                                         #(C5)
lbp_hist = {t:0 for t in range(P+2)}                                              #(C6)

for i in range(R,rowmax):                                                         #(C7)
    for j in range(R,colmax):                                                     #(C8)
        pattern = []                                                              #(C10)
        # These values are then multiplied are then multiplied by certain
        # weights and summed, leading to one single value for each
        # neighbourhood
        for p in range(P):                                                        #(C11)
            #  We use the index k to point straight down and l to point to the 
            #  right in a circular neighborhood around the point (i,j). And we 
            #  use (del_k, del_l) as the offset from (i,j) to the point on the 
            #  R-radius circle as p varies.
            del_k,del_l = R*math.cos(2*math.pi*p/P), R*math.sin(2*math.pi*p/P)    #(C12)
            if abs(del_k) < 0.001: del_k = 0.0                                    #(C13)
            if abs(del_l) < 0.001: del_l = 0.0                                    #(C14)
            k, l =  i + del_k, j + del_l                                          #(C15)
            k_base,l_base = int(k),int(l)                                         #(C16)
            delta_k,delta_l = k-k_base,l-l_base                                   #(C17)
            if (delta_k < 0.001) and (delta_l < 0.001):                           #(C18)
                image_val_at_p = float(image[k_base][l_base])                     #(C19)
            elif (delta_l < 0.001):                                               #(C20)
                image_val_at_p = (1 - delta_k) * image[k_base][l_base] +  \
                                              delta_k * image[k_base+1][l_base]   #(C21)
            elif (delta_k < 0.001):                                               #(C22)
                image_val_at_p = (1 - delta_l) * image[k_base][l_base] +  \
                                              delta_l * image[k_base][l_base+1]   #(C23)
            else:                                                                 #(C24)
                image_val_at_p = (1-delta_k)*(1-delta_l)*image[k_base][l_base] + \
                                 (1-delta_k)*delta_l*image[k_base][l_base+1]  + \
                                 delta_k*delta_l*image[k_base+1][l_base+1]  + \
                                 delta_k*(1-delta_l)*image[k_base+1][l_base]      #(C25)
            if image_val_at_p >= image[i][j]:                                     #(C26)
                pattern.append(1)                                                 #(C27)
            else:                                                                 #(C28)
                pattern.append(0)                                                 #(C29)
        print("pattern: %s" % pattern)
        
        lbp_code = 0
        
        # Aqui começo a alterar o código original...

#       print("encoding: %s" % encoding)                                          #(C48)
#print("\nLBP Histogram: %s" % lbp_hist)                                           #(C49)

