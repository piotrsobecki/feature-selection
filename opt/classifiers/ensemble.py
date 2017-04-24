import numpy
import math
from numpy import unique
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_predict


class ProbabalisticClassifier():
    def __init__(self, inner, columns=None, alpha=1, beta=0, p=0, weight=0, method="precision_recal", cv=10):
        self.inner = inner
        self.columns = columns
        self.cv = cv
        self.weight = weight
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.method = method

    def fit(self, X, Y, weighting=accuracy_score):
        if self.columns is None:
            self.columns = set(X.columns.values)
        X_ss = X.loc[:,list(self.columns)]
        self.inner.fit(X_ss, Y)
        self.precisions, self.recalls, self.weight = self.precisions_recalls_weight(X_ss, Y, weighting=weighting)
        self.labels = numpy.unique(Y)
        return self


    def p0(self,X,**kwargs):
        if self.method is 'predict_proba' and getattr(self.inner, 'predict_proba') is not None:
            yhat =  self.inner.predict_proba(X=X[list(self.columns)], **kwargs)
        else:
            yhat = self.inner.predict(X=X[list(self.columns)])
            yhat_onehot = numpy.zeros((len(X), len(self.labels)))
            yhat_onehot[numpy.arange(len(X)), yhat] = 1
            yhat = yhat_onehot
        return yhat

    def p1(self,p0):
        y = numpy.argmax(p0, axis=1)
        out = []
        recall = 0
        for pred, rec in self.recalls.items():
            recall += rec
        for pred in  y:
            p = {}
            p_sum = 0
            for prec_key in self.precisions:
                if pred == prec_key:
                    p[prec_key] = max(0, min(1, max(self.precisions[pred], self.p)))
                else:
                    p[prec_key] = (1 - self.precisions[pred]) * (1 - self.recalls[prec_key]) / ( recall - (1 - self.recalls[pred]))
                p_sum += p[prec_key]
            for prec_key in self.precisions:
                p[prec_key] = p[prec_key] / p_sum
            out.append(list(p.values()))
        return numpy.array(out)

    def precisions_recalls_weight(self, X, Y, weighting=accuracy_score):
        yhat = cross_val_predict(self.inner, X, Y, cv=self.cv)
        recalls = {}
        precisions = {}
        weight = self.weight
        if not weight:
            weight = weighting(Y, yhat)
        for label in unique(Y):
            Y_binary = Y == label
            P_binary = yhat == label
            recalls[label] = recall_score(Y_binary, P_binary)
            precisions[label] = precision_score(Y_binary, P_binary)
        return precisions, recalls, weight

    def predict(self, X, **kwargs):
        X_ss = X.loc[:,list(self.columns)]
        return self.inner.predict(X=X_ss, **kwargs)

    def predict_proba(self, X, **kwargs):
        p0 = self.p0(X,**kwargs)
        p1 = self.p1(p0)
        p = numpy.add(self.beta * p1, self.alpha * p0) + self.p
        #p = math.pow(math.e,p)/(1+math.pow(math.e,p))
        #Normalize
        row_sums = p.sum(axis=1)
        return p / row_sums[:, numpy.newaxis]

    def supports(self, X):
        return set(X.columns.values).issuperset(self.columns)


class WeightedEnsemble(BaseEstimator):
    def __init__(self, classifiers, cv=10):
        self.classifiers = classifiers
        self.cv = cv

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            self.classifiers[i].weight = weight

    def fit(self, X, Y):
        for classifier in self.classifiers:
            classifier.fit(X, Y)
        return self

    def predict(self, X):
        combined = self.predict_proba(X)
        out = []
        for case in combined:
            out.append(numpy.argmax(case))
        return out

    def predict_proba(self, X):
        predictions = []
        weights = [cls.weight for cls in self.classifiers]
        for classifier in self.classifiers:
            if classifier.supports(X):
                predictions.append(classifier.predict_proba(X))
        combined = []
        for classifier in range(len(predictions)):
            for case in range(len(predictions[classifier])):
                predictions_for_x = weights[classifier] * numpy.array(predictions[classifier][case])
                if case < len(combined):
                    combined[case] = numpy.add(combined[case], predictions_for_x)
                else:
                    combined.append(predictions_for_x)
        combined = numpy.array(combined)
        return combined / combined.sum(axis=1)[:, numpy.newaxis]
