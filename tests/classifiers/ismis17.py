import pandas

import numpy as np
from sklearn import datasets
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold, StratifiedKFold

from opt.classifiers.ensemble import ConsciousClassifier, ProbabilityEnsemble

def crossvalidation(clf, X, Y, K=10):
    score = np.zeros(K)
    kf = StratifiedKFold(n_splits=K, shuffle=True)
    i = 0
    for train, test in kf.split(X, Y):
        print(i)
        X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], Y[train], Y[test]
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        score[i] = accuracy_score(y_test, y_pred)
        i += 1
    return score




#Weighted

iris = datasets.load_iris()

X = pandas.DataFrame(iris.data, columns=iris.feature_names)
Y = iris.target

print(iris.feature_names)

cls_inner = KNeighborsClassifier(n_neighbors=5)

cls1 = ConsciousClassifier(cls_inner,columns=[iris.feature_names[0],iris.feature_names[1]])

cls2 = ConsciousClassifier(cls_inner,columns=[iris.feature_names[0],iris.feature_names[2]])

cls3 = ConsciousClassifier(cls_inner,columns=[iris.feature_names[0],iris.feature_names[3]])

cls4 = ConsciousClassifier(cls_inner,columns=[iris.feature_names[1],iris.feature_names[2]])

cls5 = ConsciousClassifier(cls_inner,columns=[iris.feature_names[1],iris.feature_names[3]])

cls6 = ConsciousClassifier(cls_inner,columns=[iris.feature_names[2],iris.feature_names[3]])

cls_ensemble = ProbabilityEnsemble([cls1,cls2,cls3,cls4,cls5,cls6])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('Ensemble 1 - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))

cls_ensemble = ProbabilityEnsemble([cls1,cls4,cls6])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('Ensemble 2 - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))


cls_ensemble = ProbabilityEnsemble([cls1,cls6])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('Ensemble 3 - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))

cls_ensemble = ProbabilityEnsemble([cls3,cls4])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('Ensemble 4 - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))




#UnWeighted

iris = datasets.load_iris()

X = pandas.DataFrame(iris.data, columns=iris.feature_names)
Y = iris.target

print(iris.feature_names)

cls_inner = KNeighborsClassifier(n_neighbors=5)

cls1 = ConsciousClassifier(cls_inner,weight=1,columns=[iris.feature_names[0],iris.feature_names[1]])

cls2 = ConsciousClassifier(cls_inner,weight=1,columns=[iris.feature_names[0],iris.feature_names[2]])

cls3 = ConsciousClassifier(cls_inner,weight=1,columns=[iris.feature_names[0],iris.feature_names[3]])

cls4 = ConsciousClassifier(cls_inner,weight=1,columns=[iris.feature_names[1],iris.feature_names[2]])

cls5 = ConsciousClassifier(cls_inner,weight=1,columns=[iris.feature_names[1],iris.feature_names[3]])

cls6 = ConsciousClassifier(cls_inner,weight=1,columns=[iris.feature_names[2],iris.feature_names[3]])

cls_ensemble = ProbabilityEnsemble([cls1,cls2,cls3,cls4,cls5,cls6])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('Ensemble 1 - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))

cls_ensemble = ProbabilityEnsemble([cls1,cls4,cls6])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('Ensemble 2 - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))


cls_ensemble = ProbabilityEnsemble([cls1,cls6])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('Ensemble 3 - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))

cls_ensemble = ProbabilityEnsemble([cls3,cls4])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('Ensemble 4 - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))

#VOTING


cls1 = ConsciousClassifier(cls_inner,weight=1,p=1,columns=[iris.feature_names[0],iris.feature_names[1]])

cls2 = ConsciousClassifier(cls_inner,weight=1,p=1,columns=[iris.feature_names[0],iris.feature_names[2]])

cls3 = ConsciousClassifier(cls_inner,weight=1,p=1,columns=[iris.feature_names[0],iris.feature_names[3]])

cls4 = ConsciousClassifier(cls_inner,weight=1,p=1,columns=[iris.feature_names[1],iris.feature_names[2]])

cls5 = ConsciousClassifier(cls_inner,weight=1,p=1,columns=[iris.feature_names[1],iris.feature_names[3]])

cls6 = ConsciousClassifier(cls_inner,weight=1,p=1,columns=[iris.feature_names[2],iris.feature_names[3]])


cls_ensemble = ProbabilityEnsemble([cls1,cls2,cls3,cls4,cls5,cls6])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('Voting 1 - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))

cls_ensemble = ProbabilityEnsemble([cls1,cls4,cls6])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('Voting 2 - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))


cls_ensemble = ProbabilityEnsemble([cls1,cls6])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('Voting 3 - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))

cls_ensemble = ProbabilityEnsemble([cls3,cls4])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('Voting 4 - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))




scores = crossvalidation(cls_inner,X,Y,3)

print('KNN - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))

