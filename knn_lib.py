# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 18:21:42 2021

@author: user
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt;
iris_dataset=load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0)

kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(X_train, y_train);

prediction = kn.predict(X_test);

import sklearn.metrics as sm;

print("ACCURACY SCORE:",sm.accuracy_score(y_test,prediction));
print("CONFUSION MATRIX:\n",sm.confusion_matrix(y_test,prediction));

plt.plot(X_test,y_test,'ro');
plt.plot(X_test,prediction,'b+');

