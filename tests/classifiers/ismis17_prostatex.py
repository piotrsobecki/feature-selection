import pandas

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold, StratifiedKFold

from opt.classifiers.ensemble import ProbabalisticClassifier, WeightedEnsemble
from sklearn.metrics import roc_curve, auc


def roc_auc_score(y_test, y_score):
 fpr, tpr, _ = roc_curve(y_test, y_score)
 return auc(fpr, tpr)


def prostatex_auc(labels, predictions):
 return roc_auc_score(labels, predictions[:, 1])


def crossvalidation(clf, X, Y, K=3):
 kf = StratifiedKFold(n_splits=K, random_state=0)
 y_proba = cross_val_predict(clf, X, Y, cv=kf, method='predict_proba')
 return prostatex_auc(Y, y_proba)


# Weighted



X = pandas.DataFrame.from_csv("tests/data/PROSTATEx/features.csv", sep=";", index_col=None)
Y = X.loc[:, 'clinsig'].values
del X['clinsig']

T2_COR =  [attr for attr in X.columns if "modality=t2-cor" in attr]
T2_SAG =  [attr for attr in X.columns if "modality=t2-sag" in attr]
T2_TRA =  [attr for attr in X.columns if "modality=t2-tra" in attr]
DWI_ADC = [attr for attr in X.columns if "modality=dwi-adc" in attr]
KTRANS =  [attr for attr in X.columns if "modality=ktrans" in attr]

T2_ALL = [*T2_COR, *T2_SAG, *T2_TRA]

LOCATION = [attr for attr in X.columns if "in_zone" in attr]

T2_COR.extend(LOCATION)
T2_SAG.extend(LOCATION)
T2_TRA.extend(LOCATION)
T2_ALL.extend(LOCATION)
DWI_ADC.extend(LOCATION)
KTRANS.extend(LOCATION)

n_neighbors=7

def test(name, alpha, beta, weight, method='predict_proba'):
    t2_cls = ProbabalisticClassifier(KNeighborsClassifier(n_neighbors=n_neighbors), method=method, alpha=alpha, beta=beta,  weight=weight, columns=T2_ALL)
    t2cor_cls = ProbabalisticClassifier(KNeighborsClassifier(n_neighbors=n_neighbors), method=method, alpha=alpha, beta=beta,  weight=weight, columns=T2_COR)
    t2sag_cls = ProbabalisticClassifier(KNeighborsClassifier(n_neighbors=n_neighbors), method=method, alpha=alpha, beta=beta, weight=weight, columns=T2_SAG)
    t2tra_cls = ProbabalisticClassifier(KNeighborsClassifier(n_neighbors=n_neighbors), method=method, alpha=alpha, beta=beta, weight=weight, columns=T2_TRA)
    dwiadc_cls = ProbabalisticClassifier(KNeighborsClassifier(n_neighbors=n_neighbors), method=method, alpha=alpha, beta=beta, weight=weight, columns=DWI_ADC)
    ktrans_cls = ProbabalisticClassifier(KNeighborsClassifier(n_neighbors=n_neighbors), method=method, alpha=alpha, beta=beta, weight=weight, columns=KTRANS)
    if weight is None:
        weight = 0

    def test_ensemble(subname,cls):

        cls_ensemble = WeightedEnsemble(cls)
        score = crossvalidation(cls_ensemble, X, Y, 3)
        print('%s; %s ;%0.2f;%0.2f;%0.2f; %0.2f ' % (name,subname,alpha,beta,weight,score))

    test_ensemble("ALL",[t2cor_cls, t2sag_cls, t2tra_cls, dwiadc_cls, ktrans_cls])
    test_ensemble("T2-TRA+ADC",[t2tra_cls, dwiadc_cls])
    test_ensemble("T2",[t2tra_cls, t2sag_cls, t2cor_cls])
    test_ensemble("DWI+KTRANS",[dwiadc_cls, ktrans_cls])
    test_ensemble("T2+DWI+KTRANS",[t2_cls,dwiadc_cls, ktrans_cls])
    test_ensemble("T2+DWI",[t2_cls,dwiadc_cls])
    test_ensemble("T2+KTRANS",[t2_cls,dwiadc_cls])



# Weighted
print("Configuration;Version;Alpha;Beta;Weight;AUC;Err")
test("Probabalistic Ensemble",   0,   1, None)
test("Probabalistic Ensemble", .50, .50, None)
test("Probabalistic Ensemble", .25, .75, None)
test("Probabalistic Ensemble", .75, .25, None)

# UnWeighted
test("Probabalistic Ensemble",   0,   1, 1)
test("Probabalistic Ensemble", .50, .50, 1)
test("Probabalistic Ensemble", .25, .75, 1)
test("Probabalistic Ensemble", .75, .25, 1)

# VOTING
test("Voting Ensemble", 1, 0, None)
test("Voting Ensemble", 1, 0, 1)


score = crossvalidation(KNeighborsClassifier(n_neighbors=n_neighbors), X, Y, 3)

print('KNN - Accuracy: %0.2f ' % ( score))
