#has the required libraries

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
#need only these categories
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med'];
#this takes the data from the train column
twenty_train = fetch_20newsgroups(subset='train',categories=categories,shuffle=True)
twenty_test = fetch_20newsgroups(subset='test',categories=categories,shuffle=True)
#the twenty train and the twenty_test are in the form of the dicionary
print(len(twenty_train.data))
print(len(twenty_test.data))
#access the target names
print(twenty_train.target_names)
print("\n".join(twenty_train.data[0].split("\n")))
print("****",twenty_train.target[0])#train data first column
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_tf = count_vect.fit_transform(twenty_train.data);
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_tf)
##print(X_train_tfidf.shape)
from sklearn.naive_bayes import MultinomialNB
#output from the tfdf id the input for multinomial NB
from sklearn.metrics import accuracy_score
from sklearn import metrics
mod = MultinomialNB();
#learn the machine -> training the model
mod.fit(X_train_tfidf, twenty_train.target)
X_test_tf = count_vect.transform(twenty_test.data)
X_test_tfidf = tfidf_transformer.transform(X_test_tf)
print(X_test_tfidf.shape)
predicted = mod.predict(X_test_tfidf)
print(predicted)
print("Accuracy:", accuracy_score(twenty_test.target, predicted));
print(classification_report(twenty_test.target,predicted,target_names=twenty_test.target_names))

print("confusion matrix is \n",metrics.confusion_matrix(twenty_test.target, predicted))
