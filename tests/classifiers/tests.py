import pandas

from sklearn import datasets
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

from opt.classifiers.ensemble import ConsciousClassifier, ProbabilityEnsemble

iris = datasets.load_iris()

X = pandas.DataFrame(iris.data, columns=iris.feature_names)
Y = iris.target

cls1 = ConsciousClassifier(KNeighborsClassifier())
cls2 = ConsciousClassifier(KNeighborsClassifier(n_neighbors=1))
cls3 = ConsciousClassifier(KNeighborsClassifier(n_neighbors=5))
cls4 = ConsciousClassifier(KNeighborsClassifier(n_neighbors=3))

cls1.fit(X[[iris.feature_names[0],iris.feature_names[1]]],Y)
cls2.fit(X[[iris.feature_names[1],iris.feature_names[2]]],Y)
cls3.fit(X[[iris.feature_names[2],iris.feature_names[3]]],Y)
cls4.fit(X,Y)

cls_ensemble = ProbabilityEnsemble([cls1, cls2, cls3, cls4])
yhat = cross_val_predict(cls_ensemble, X, Y, cv=10)
for label in pandas.unique(Y):
    Y_binary = Y == label
    P_binary = yhat == label
    print(label)
    print(recall_score(Y_binary, P_binary))
    print(precision_score(Y_binary, P_binary))
#cls_ensemble.fit(X,Y)