import json, csv
import array, random, numpy as np
from conda._vendor.auxlib.configuration import Configuration
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from opt.genetic import GeneticOptimizer
from deap import creator, base, tools, algorithms


class RoutingHOF():
    def __init__(self, optimizer, context):
        self.optimizer = optimizer
        self.context = context
        self.ngen = 0
        self.results = None

    def insert(self, item):
        pass

    def update(self, population):
        results = self.optimizer.results([self.optimizer.configuration(x) for x in population])
        self.optimizer.on_gen_end(self.context, self.ngen, results)
        self.ngen += 1
        self.results = results


class FeatureSelectionConfiguration(Configuration):
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
    def __init__(self, clfs, features, labels, score_func, **settings):
        self.clfs = clfs
        self.features = features
        self.labels = labels.as_matrix()
        self.score_func = score_func
        super().__init__(**settings)

    def configuration(self, individual):
        return FeatureSelectionConfiguration(individual, self.features.columns)

    def default_settings(self):
        defaults = super().default_settings()
        defaults['cv_fold'] = 3
        defaults['str(clf)'] = str(self.clfs)
        defaults['n'] = self.features_len()
        return defaults

    def features_len(self):
        return self.features.shape[1]

    def individual(self, toolbox):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, self.features_len())

    def eval(self, individual):
        fitness = [0]
        columns = self.configuration(individual).columns()
        if len(columns) > 0:
            features_subset = self.features.as_matrix(columns=columns)
            for clf in self.clfs:
                y_proba = cross_val_predict(clf, features_subset, self.labels,
                                            cv=StratifiedKFold(n_splits=self.cv_fold, random_state=0),
                                            method='predict_proba')
                fitness.append(self.score_func(self.labels, y_proba))
        else:
            fitness = np.zeros(len(self.clfs))
        return max(fitness),

    def on_gen_end(self, context, generation_no, results):
        super().on_gen_end(context, generation_no, results)
        max_config = results.max()
        row = [generation_no, max_config.value()]
        row.extend(max_config.as_list())
        gen_row = [generation_no]
        gen_row.extend(results.fitness())
        self.write_row_2file(row, context['csv'], context['csv_file'])
        self.write_row_2file(gen_row, context['csv_gen'], context['csv_gen_file'])

    def write_row_2file(self, row, csv_writer, file):
        csv_writer.writerow(row)
        file.flush()

    def write_gen_row(self, row, context):
        context['csv_gen'].writerow(row)
        context['csv_gen_file'].flush()

    def on_fit_start(self, context):
        super().on_fit_start(context)
        cols = ['Generation', 'Max Fitness']
        cols.extend(self.features.columns.tolist())
        gencols = ['Generation']
        gencols.extend(['#' + str(x) for x in range(0, self.n)])
        context['csv_file'] = open(self.datalog, 'a+')
        context['csv_gen_file'] = open(self.genlog, 'a+')
        context['csv'] = csv.writer(context['csv_file'], delimiter=';', lineterminator='\n')
        context['csv_gen'] = csv.writer(context['csv_gen_file'], delimiter=';', lineterminator='\n')
        self.write_row_2file(cols, context['csv'], context['csv_file'])
        self.write_row_2file(gencols, context['csv_gen'], context['csv_gen_file'])

    def on_fit_end(self, context):
        super().on_fit_end(context)
        context['csv_file'].close()
        context['csv_gen_file'].close()
