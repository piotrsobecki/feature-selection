import json, csv
import array, random, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict,cross_val_score

from opt.genetic import GeneticOptimizer, GeneticConfiguration
from deap import creator, base, tools, algorithms

class FeatureSelectionConfiguration(GeneticConfiguration):
    def __init__(self, individual, all_columns):
        super().__init__(individual)
        self.all_columns = all_columns

    def column_indices(self):
        return [i for i, j in enumerate(self.individual) if j]

    # As list of active and not active columns
    def as_list(self):
        return [v for v in self.individual]

    def columns(self):
        return self.all_columns[self.column_indices()]

    def __str__(self):
        cols = self.columns()
        return json.dumps({
            'fitness': self.value(),
            'columns_length': len(cols),
            'columns_str': str(cols.tolist()),
            'indices_str': str(self.individual)
        }, indent=4, sort_keys=True)


class CVGeneticFeatureSelection(GeneticOptimizer):
    def __init__(self, clfs, features, labels, score_func=None, **settings):
        self.clfs = clfs
        self.features = features
        self.labels = labels.as_matrix()
        self.score_func = score_func
        super().__init__(**settings)

    def configuration(self, individual):
        return FeatureSelectionConfiguration(individual, self.features.columns)

    def default_settings(self):
        return {
            **super().default_settings(),
            "cv_fold": 3,
            "str(clf)":str(self.clfs),
            "n": self.features_len()
        }

    def features_len(self):
        return self.features.shape[1]

    def individual(self, toolbox):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, self.features_len())

    def eval_on(self,clfs,features,labels):
        fitness = [0]
        cv = StratifiedKFold(n_splits=self.cv_fold, random_state=0)
        for clf in clfs:
            if self.score_func is not None:
                y_proba = cross_val_predict(clf, features, labels, cv=cv, method='predict_proba')
                fitness.append(self.score_func(labels, y_proba))
            else:
                fitness.append(cross_val_score(clf, features, labels,  cv=cv).mean())
        return max(fitness)

    def eval(self, individual):
        fitness = 0
        columns = self.configuration(individual).columns()
        if len(columns) > 0:
            features_subset = self.features.as_matrix(columns=columns)
            fitness = self.eval_on(self.clfs,features_subset,self.labels)
        return fitness,