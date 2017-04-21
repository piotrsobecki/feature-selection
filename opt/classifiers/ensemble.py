import numpy
from numpy import unique
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_predict


class ConsciousClassifier():
    def __init__(self, inner, columns=None, p =0, weight=0, cv=10):
        self.inner = inner
        self.columns = columns
        self.cv = cv
        self.weight = weight
        self.p=p

    def fit(self, X, Y):
        if self.columns is None:
            self.columns = set(X.columns.values)
        else:
            X = X[list(self.columns)]
        self.inner.fit(X, Y)
        yhat = cross_val_predict(self.inner, X, Y, cv=self.cv)
        self.recalls = {}
        self.precisions = {}
        if not self.weight:
            self.weight = accuracy_score(Y, yhat)
        for label in unique(Y):
            Y_binary = Y == label
            P_binary = yhat == label
            self.recalls[label] = recall_score(Y_binary, P_binary)
            self.precisions[label] = precision_score(Y_binary, P_binary)
        return self

    def predict(self, X, **kwargs):
        return self.inner.predict(X=X[list(self.columns)], **kwargs)

    def predict_proba(self, X, **kwargs):
        # if getattr(self.inner, 'predict_proba') is not None:
        #    return self.inner.predict_proba(X=X[list(self.columns)], **kwargs)
        out = []
        recall = 0
        for pred, rec in self.recalls.items():
            recall += rec
        for pred in self.predict(X):
            p = {}
            for prec_key in self.precisions:
                if pred == prec_key:
                    p[prec_key] = max(0,min(1, max(self.precisions[pred], self.p)))
                else:
                    p[prec_key] = (1 - self.precisions[pred]) * (1 - self.recalls[prec_key]) / (recall - (1 - self.recalls[pred]))
            out.append(list(p.values()))
        return out

    def supports(self, X):
        return set(X.columns.values).issuperset(self.columns)


class ProbabilityEnsemble(BaseEstimator):
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
