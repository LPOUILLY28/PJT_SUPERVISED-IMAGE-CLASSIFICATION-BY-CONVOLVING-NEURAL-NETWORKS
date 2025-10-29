# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:44:46 2020

@author: eric
"""
import numpy as np
import sklearn.datasets as ds
import sklearn.svm as svm

c=1
algo = svm.LinearSVC(C = c,max_iter=1000)

""
decision_function='ovo' # for one versus one classification
decision_function='ovr' # for one versus rest classification par defaut
algo = svm.SVC(C = c,max_iter=1000,decision_function_shape=decision_function)
""
#entrainement


algo.fit(train_data,train_target)

#prediction
test_predictions=algo.predict(test_data)

iris = ds.load_iris()

features_iris = iris.data
target_iris = iris.target
#mélange des indices
randomized_indices = np.arange(len(features_iris))
np.random.shuffle(randomized_indices)
#mélange des données
shuffled_features_iris = features_iris[randomized_indices]
shuffled_target_iris = target_iris[randomized_indices]


#The Iris datset 
#each training feature is composed as follow:
# 1. sepal length in cm 
# 2. sepal width in cm 
# 3. petal length in cm 
# 4. petal width in cm
#classes are:
# 0 : Iris Setosa 
# 1 : Iris Versicolour 
# 2 : Iris Virginica