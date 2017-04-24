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

from opt.classifiers.ensemble import ProbabalisticClassifier, WeightedEnsemble

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



X = pandas.DataFrame.from_csv("tests/data/PROSTATEx/features.csv",sep=";",index_col=None)
Y = X.loc[:,'clinsig'].values
del X['clinsig']



T2_COR =    [attr for attr in X.columns if "modality=t2-cor" in attr]
T2_SAG =    [attr for attr in X.columns if "modality=t2-sag" in attr]
T2_TRA =    [attr for attr in X.columns if "modality=t2-tra" in attr]
DWI_ADC =   [attr for attr in X.columns if "modality=dwi-adc" in attr]
KTRANS =    [attr for attr in X.columns if "modality=ktrans" in attr]

LOCATION = [attr for attr in X.columns if "in_zone" in attr]

T2_COR.extend(LOCATION)
T2_SAG.extend(LOCATION)
T2_TRA.extend(LOCATION)
DWI_ADC.extend(LOCATION)
KTRANS.extend(LOCATION)

t2cor_cls = ProbabalisticClassifier(KNeighborsClassifier(n_neighbors=5), columns=T2_COR)

t2sag_cls = ProbabalisticClassifier(KNeighborsClassifier(n_neighbors=5), columns=T2_SAG)

t2tra_cls = ProbabalisticClassifier(KNeighborsClassifier(n_neighbors=5), columns=T2_TRA)

dwiadc_cls = ProbabalisticClassifier(KNeighborsClassifier(n_neighbors=5), columns=DWI_ADC)

ktrans_cls = ProbabalisticClassifier(KNeighborsClassifier(n_neighbors=5), columns=KTRANS)

cls_ensemble = WeightedEnsemble([t2cor_cls, t2sag_cls, t2tra_cls, dwiadc_cls, ktrans_cls])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('Ensemble 1 ALL - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))

cls_ensemble = WeightedEnsemble([t2tra_cls, dwiadc_cls])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('Ensemble 2 T2-TRA ADC - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))


cls_ensemble = WeightedEnsemble([t2tra_cls,t2sag_cls,t2cor_cls])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('Ensemble 3 T2 - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))

cls_ensemble = WeightedEnsemble([dwiadc_cls,ktrans_cls])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('Ensemble 4 DWI KTRANS- Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))


#UnWeighted


t2cor_cls = ProbabalisticClassifier(KNeighborsClassifier(n_neighbors=5), weight=1, columns=T2_COR)

t2sag_cls = ProbabalisticClassifier(KNeighborsClassifier(n_neighbors=5), weight=1, columns=T2_SAG)

t2tra_cls = ProbabalisticClassifier(KNeighborsClassifier(n_neighbors=5), weight=1, columns=T2_TRA)

dwiadc_cls = ProbabalisticClassifier(KNeighborsClassifier(n_neighbors=5), weight=1, columns=DWI_ADC)

ktrans_cls = ProbabalisticClassifier(KNeighborsClassifier(n_neighbors=5), weight=1, columns=KTRANS)


cls_ensemble = WeightedEnsemble([t2cor_cls, t2sag_cls, t2tra_cls, dwiadc_cls, ktrans_cls])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('Ensemble 1 ALL - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))

cls_ensemble = WeightedEnsemble([t2tra_cls, dwiadc_cls])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('Ensemble 2 T2-TRA ADC - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))


cls_ensemble = WeightedEnsemble([t2tra_cls,t2sag_cls,t2cor_cls])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('Ensemble 3 T2 - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))

cls_ensemble = WeightedEnsemble([dwiadc_cls,ktrans_cls])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('Ensemble 4 DWI KTRANS- Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))

#VOTING


t2cor_cls = ProbabalisticClassifier(KNeighborsClassifier(n_neighbors=5), weight=1, p=1, columns=T2_COR)

t2sag_cls = ProbabalisticClassifier(KNeighborsClassifier(n_neighbors=5), weight=1, p=1, columns=T2_SAG)

t2tra_cls = ProbabalisticClassifier(KNeighborsClassifier(n_neighbors=5), weight=1, p=1, columns=T2_TRA)

dwiadc_cls = ProbabalisticClassifier(KNeighborsClassifier(n_neighbors=5), weight=1, p=1, columns=DWI_ADC)

ktrans_cls = ProbabalisticClassifier(KNeighborsClassifier(n_neighbors=5), weight=1, p=1, columns=KTRANS)


cls_ensemble = WeightedEnsemble([t2cor_cls, t2sag_cls, t2tra_cls, dwiadc_cls, ktrans_cls])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('VOTING 1 ALL - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))

cls_ensemble = WeightedEnsemble([t2tra_cls, dwiadc_cls])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('VOTING 2 T2-TRA ADC - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))


cls_ensemble = WeightedEnsemble([t2tra_cls,t2sag_cls,t2cor_cls])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('VOTING 3 T2 - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))

cls_ensemble = WeightedEnsemble([dwiadc_cls,ktrans_cls])
scores = crossvalidation(cls_ensemble,X,Y,3)
#cls_ensemble.fit(X,Y)

print('VOTING 4 DWI KTRANS- Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))

scores = crossvalidation(KNeighborsClassifier(n_neighbors=5),X,Y,3)

print('KNN - Accuracy: %0.2f (+/- %0.2f)'%(np.mean(scores),scores.std()*2))

