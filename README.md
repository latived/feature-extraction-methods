# feature-extraction-methods

Vejamos... em cada pasta há os códigos que fiz para replicar uns trabalhos de
extração e classificação de atributos a partir de imagens médicas.

## 01: _Texture-based Polyp Detection in Colonoscopy_ {[DOI](https://doi.org/10.1007/978-3-540-93860-6_70)}

### Atributos

* GLCM 16: Four GLCM were computed for each patch with d = 1 and 
α ∈ {0◦, 45◦, 90◦, 135◦}. The following features have been computed on 
each matrix, leading to 16-dimensional feature vectors: 
energy, homogeneity, entropy, and correlation.

* GLCM 6: Four GLCM were computed as described above. The extracted features 
were: energy, homogeneity, entropy, inertia, cluster shade and cluster 
prominence. The means of the four values (from the four matrices) of features 
were: energy, homogeneity, entropy, inertia, cluster shade and clus-ter prominence.

* LBP: The LBP is extracted from a 3×3 neighbourhood as follows: 
The neighbour-hood is binarised using the value of the center pixel as threshold. 
These values are then multiplied by certain weights and summed, leading to 
one single value for each neighbourhood. LBP feature vectors are formed by 
histogram bins of the distribution of the LBP values. An amount of 64 
histogram bins was used leading to a 64-dimensional feature vector.

* OC-LBP: The OC-LBP feature was extracted with an amount of 64 bins for each 
histogram, leading to a 576-dimensional feature vector

### Classificação

_With the extracted feature data an SVM-classifier was trained and tested 
using a radial basis function as kernel. We applied a stratified k-fold 
cross-validation with k = 4._

## 02: ...
