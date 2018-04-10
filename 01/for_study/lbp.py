#!/usr/bin/env python

import random
import math
#import BitVector

import numpy as np

# Vejamos, no artigo original temos (TBPDIC):
#   .. an amount of 64 histogram bins was used
#   leading to a 64-dimensional feature vector.
# Num patch 50x50, usando uma matriz 3x3 temos para cada matriz:
#   LBP_P,R (x_c, y_c) = sum p = 0 até P - 1 de s(g_p - g_c)2^p 
#      LBP_P,R resulta em 2^P valores distintos
#   Nessa matriz 3x3 temos P = 8, logo 2^8 = 256 valores   
#   Porém... no artigo o histograma tem 64 bins, não 256. 

# recebe image 50x50 (patch)
# retorna 48^2 lbp_codes
def lbp(image):
    IMAGE_SIZE = 50                                                                     
    GRAY_LEVELS = 256                                                                    
    R = 1                  # the parameter R is radius of the circular pattern         
    P = 8                  # the number of points to sample on the circle              

    # lbp = [[0 for _ in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)]          
     
    rowmax,colmax = IMAGE_SIZE-R,IMAGE_SIZE-R                                          
    lbp_hist = {t:0 for t in range(GRAY_LEVELS)} 
    
    for i in range(R,rowmax):                                                          
        for j in range(R,colmax):                                                      
            pattern = []                                                               
            # These values are then multiplied by certain weights and summed,
            # leading to one single value for each neighbourhood

            for p in range(P):                                                         
                #  We use the index k to point straight down and l to point to the 
                #  right in a circular neighborhood around the point (i,j). And we 
                #  use (del_k, del_l) as the offset from (i,j) to the point on the 
                #  R-radius circle as p varies.
                del_k,del_l = R*math.cos(2*math.pi*p/P), R*math.sin(2*math.pi*p/P)     
                if abs(del_k) < 0.001: del_k = 0.0                                     
                if abs(del_l) < 0.001: del_l = 0.0                                     
                k, l =  i + del_k, j + del_l                                           
                k_base,l_base = int(k),int(l)                                          
                delta_k,delta_l = k-k_base,l-l_base                                    
                if (delta_k < 0.001) and (delta_l < 0.001):                            
                    image_val_at_p = float(image[k_base][l_base])                      
                elif (delta_l < 0.001):                                                
                    image_val_at_p = (1 - delta_k) * image[k_base][l_base] +  \
                                                delta_k * image[k_base+1][l_base]    
                elif (delta_k < 0.001):                                                
                    image_val_at_p = (1 - delta_l) * image[k_base][l_base] +  \
                                                delta_l * image[k_base][l_base+1]    
                else:                                                                  
                    image_val_at_p = (1-delta_k)*(1-delta_l)*image[k_base][l_base] + \
                                    (1-delta_k)*delta_l*image[k_base][l_base+1]  + \
                                    delta_k*delta_l*image[k_base+1][l_base+1]  + \
                                    delta_k*(1-delta_l)*image[k_base+1][l_base]       
                if image_val_at_p >= image[i][j]:                                      
                    pattern.append(1)                                                  
                else:                                                                  
                    pattern.append(0)                                                  
    
            #   LBP_P,R (x_c, y_c) = sum p = 0 até P - 1 de s(g_p - g_c)2^p 
            lbp_code = np.add.reduce([pattern[i] * 2**i for i in range(P)])
            lbp_hist[lbp_code] += 1        
    
    return lbp_hist
