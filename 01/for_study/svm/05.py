#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by lativ on 06/08/18 at 18:15
"""

# Conventions: type casting
import numpy as np
from sklearn import random_projection

rng = np.random.RandomState(0)
X = rng.rand(10, 2000)
X = np.array(X, dtype='float32')
X.dtype

transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
X_new.dtype

from sklearn import datasets
from sklearn.svm import SVC

iris = datasets.load_iris()
clf = SVC()

clf.fit(iris.data, iris.target)
list(clf.predict(iris.data[:3]))

clf.fit(iris.data, iris.target_names[iris.target])
list(clf.predict(iris.data[:3]))

# Refitting and updating parameters

rng = np.random.RandomState(0)
X = rng.rand(100, 10)
y = rng.binomial(1, 0.5, 100)
X_test = rng.rand(5, 10)

clf = SVC()
clf.set_params(kernel='linear').fit(X, y)

clf.predict(X_test)

clf.set_params(kernel='rbf').fit(X, y)
clf.predict(X_test)

# Multiclass vs. multlabel fitting
# Don't...