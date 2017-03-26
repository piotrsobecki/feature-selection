import logging
import json, csv
import pandas as pd
from deap import creator, base, tools, algorithms
from opt.base import Configuration, Optimizer, Results


# TODO MOVE DATALOG SOMEWHERE LOWER
class GeneticLogHelper():
    def __init__(self, genlog, datalog, sep):
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


class RoutingHOF():
    def __init__(self, optimizer, context, results_class):
        self.optimizer = optimizer
        self.context = context
        self.results_class = results_class
        self.ngen = 0
        self.results = None

    def insert(self, item):
        pass

    def update(self, population):
        results = self.results_class([self.optimizer.configuration(x) for x in population])
        self.optimizer.on_gen_end(self.context, self.ngen, results)
        self.ngen += 1
        self.results = results


class GeneticConfiguration(Configuration):
    def value(self):
        return self.individual.fitness.values[0]


# noinspection PyTypeChecker,PyUnresolvedReferences
class GeneticOptimizer(Optimizer):
    results_class = Results

    def __init__(self, **settings):
        if settings is None:
            settings = {}
        self.logger = logging.getLogger('optimizer')
        self.settings = {**self.default_settings(), **settings}

    def default_settings(self):
        return {
            'tournsize': 3,
            'indpb': 0.05,
            'ngen': 40,
            'n': 300,
            'mutpb': 0.1,
            'cxpb': 0.5,
            "fileverbose": True
        }

    def eval(self, individual):
        raise NotImplementedError()

    def individual(self, toolbox):
        raise NotImplementedError()

    def configuration(self, individual):
        return GeneticConfiguration(individual)

    def get_genlog(self):
        return pd.read_csv(self.settings['genlog'], self.settings['sep'], index_col=0)

    def get_datalog(self):
        return pd.read_csv(self.settings['datalog'], self.settings['sep'], index_col=0)

    def mate(self, toolbox):
        toolbox.register("mate", tools.cxTwoPoint)

    def mutate(self, toolbox):
        toolbox.register("mutate", tools.mutFlipBit, indpb=self.indpb)

    def evaluate(self, toolbox):
        toolbox.register("evaluate", self.eval)

    def select(self, toolbox):
        toolbox.register("select", tools.selTournament, tournsize=self.tournsize)

    def population(self, toolbox):
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def __getattr__(self, item):
        if item in self.settings:
            return self.settings[item]
        return None

    def on_gen_end(self, context, generation_no, results):
        self.logger.debug('Generation %d, max: %s' % (generation_no, results.max()))
        if self.fileverbose:
            context['log'].log(context, generation_no, results)

    def on_fit_start(self, context):
        if self.fileverbose:
            context['log'].setup(context)

    def on_fit_end(self, context):
        if self.fileverbose:
            context['log'].close(context)

    def log_helper(self):
        return GeneticLogHelper(self.settings['genlog'],self.settings['datalog'], self.settings['sep'])

    def fit(self):
        self.logger.info(self.settings)
        toolbox = base.Toolbox()
        self.individual(toolbox)
        self.population(toolbox)
        self.evaluate(toolbox)
        self.mate(toolbox)
        self.mutate(toolbox)
        self.select(toolbox)
        population = toolbox.population(self.n)
        context = {
            'settings': self.settings,
            'features': self.features,
            'log': self.log_helper()
        }
        self.on_fit_start(context)
        hof = RoutingHOF(self, context, results_class=self.results_class)
        algorithms.eaSimple(population, toolbox, cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.ngen, halloffame=hof,
                            verbose=self.verbose)
        self.on_fit_end(context)
        return hof.results

    def __str__(self):
        return "%s(Settings = %s)" % (type(self).__name__, json.dumps(self.settings, indent=4, sort_keys=True))
