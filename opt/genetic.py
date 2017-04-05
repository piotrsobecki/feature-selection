import csv
import json
import logging

import pandas as pd
from deap import base, tools, algorithms

from opt.base import Configuration, Optimizer, Results


class LogHelper():

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('optimizer')

    def log(self, context, generation_no, results):
        pass

    def setup(self, context):
        pass

    def close(self, context):
        pass

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
            "verbose": True
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
        if self.verbose:
            context['log'].log(context, generation_no, results)

    def on_fit_start(self, context):
        if self.verbose:
            context['log'].setup(context)

    def on_fit_end(self, context):
        if self.verbose:
            context['log'].close(context)

    def log_helper(self):
        return LogHelper()

    def fit(self):
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
