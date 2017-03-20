import logging
import pandas as pd
from deap import creator, base, tools, algorithms
from opt.base import Configuration, Optimizer


class GeneticConfiguration(Configuration):
    def value(self):
        return self.individual.fitness.values[0]


# noinspection PyTypeChecker,PyUnresolvedReferences
class GeneticOptimizer(Optimizer):
    def __init__(self, **settings):
        if settings is None:
            settings = {}
        self.logger = logging.getLogger('prostatex')
        self.settings = {**self.default_settings(), **settings}
        self.verbose = False

    def default_settings(self):
        return {
            'tournsize': 3,
            'indpb': 0.05,
            'ngen': 40,
            'n': 300,
            'mutpb': 0.1,
            'cxpb': 0.5
        }

    def configuration(self, individual):
        return GeneticConfiguration(individual)

    def individual(self, toolbox):
        raise NotImplementedError()
        # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)
        # toolbox.register("attr_bool", random.randint, 0, 1)
        # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 10)

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

    def eval(self, individual):
        raise NotImplementedError()
        # return sum(individual),

    def results(self, population):
        return Results(population)

    def __getattr__(self, item):
        return self.settings[item]

    def on_fit_start(self, context):
        pass

    def on_fit_end(self, context):
        pass

    def on_gen_end(self, context, gen, results):
        self.logger.debug('Generation %d, max: %s' % (gen, results.max()))

    def fit(self):
        self.logger.info(self)
        toolbox = base.Toolbox()
        self.individual(toolbox)
        self.population(toolbox)
        self.evaluate(toolbox)
        self.mate(toolbox)
        self.mutate(toolbox)
        self.select(toolbox)
        population = toolbox.population(self.n)
        context = {}
        self.on_fit_start(context)
        hof = RoutingHOF(self, context)
        algorithms.eaSimple(population, toolbox, cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.ngen, halloffame=hof,
                            verbose=self.verbose)
        self.on_fit_end(context)
        return hof.results

    def __str__(self):
        return "%s(Settings = %s)" % (type(self).__name__, json.dumps(self.settings, indent=4, sort_keys=True))
