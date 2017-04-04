import json, csv
import array, random, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict, cross_val_score

from opt.genetic import GeneticOptimizer, GeneticConfiguration, LogHelper
from deap import creator, base, tools


class GeneticLogHelper(LogHelper):
    def __init__(self, genlog, datalog, sep):
        super().__init__()
        self.genlog = genlog
        self.datalog = datalog
        self.sep = sep

    def get_genlog(self):
        return pd.read_csv(self.genlog, self.sep, index_col=0)

    def get_datalog(self):
        return pd.read_csv(self.datalog, self.sep, index_col=0)

    def write_row_2file(self, row, csv_writer, file):
        csv_writer.writerow(row)
        file.flush()

    def log(self, context, generation_no, results):
        self.log_generation(context, generation_no, results)
        self.log_configuration(context, generation_no, results)

    def log_generation(self, context, generation_no, results):
        gen_row = [generation_no]
        gen_row.extend(results.fitness())
        self.write_row_2file(gen_row, context['csv_gen'], context['csv_gen_file'])

    def log_configuration(self, context, generation_no, results):
        max_config = results.max()
        row = [generation_no, max_config.value()]
        row.extend(max_config.as_list())
        self.write_row_2file(row, context['csv'], context['csv_file'])

    def setup_genlog(self, context):
        gencols = ['Generation']
        gencols.extend(['#' + str(x) for x in range(0, context['settings']['n'])])
        context['csv_gen_file'] = open(self.genlog, 'a+')
        context['csv_gen'] = csv.writer(context['csv_gen_file'], delimiter=';', lineterminator='\n')
        self.write_row_2file(gencols, context['csv_gen'], context['csv_gen_file'])

    def setup_configuration_log(self, context):
        cols = ['Generation', 'Max Fitness']
        cols.extend(context['features'].columns.tolist())
        context['csv_file'] = open(self.datalog, 'a+')
        context['csv'] = csv.writer(context['csv_file'], delimiter=';', lineterminator='\n')
        self.write_row_2file(cols, context['csv'], context['csv_file'])

    def setup(self, context):
        self.setup_configuration_log(context)
        self.setup_genlog(context)

    def close(self, context):
        context['csv_file'].close()
        context['csv_gen_file'].close()

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
        self.labels = labels[labels.columns[0]].tolist()
        self.score_func = score_func
        super().__init__(**settings)
        self.settings['n'] = min(self.settings['n'], self.settings['n_max'])

    def configuration(self, individual):
        return FeatureSelectionConfiguration(individual, self.features.columns)

    def default_settings(self):
        return {
            **super().default_settings(),
            "cv_fold": 3,
            "str(clf)": str(self.clfs),
            "n": self.features_len(),
            "n_max": 1000
        }
    def log_helper(self):
        return GeneticLogHelper(self.settings['genlog'],self.settings['datalog'], self.settings['sep'])

    def features_len(self):
        return self.features.shape[1]

    def individual(self, toolbox):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, self.features_len())

    def eval_on(self, clfs, features, labels):
        fitness = [0]
        cv = StratifiedKFold(n_splits=self.cv_fold, random_state=0)
        for clf in clfs:
            if self.score_func is not None:
                y_proba = cross_val_predict(clf, features, labels, cv=cv, method='predict_proba')
                fitness.append(self.score_func(labels, y_proba))
            else:
                fitness.append(cross_val_score(clf, features, labels, cv=cv).mean())
        return max(fitness)

    def eval(self, individual):
        fitness = 0
        columns = self.configuration(individual).columns()
        if len(columns) > 0:
            features_subset = self.features.as_matrix(columns=columns)
            fitness = self.eval_on(self.clfs, features_subset, self.labels)
        return fitness,
