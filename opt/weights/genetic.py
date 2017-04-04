import random

import array
from opt.genetic import GeneticOptimizer, LogHelper
from deap import creator, base, tools, algorithms


class WeightsLogHelper(LogHelper):
    def log(self, context, generation_no, results):
        config = results.max()
        self.logger.log('Generation %d: %s' % (generation_no, config))


class WeightOptimizer(GeneticOptimizer):
    def individual(self, toolbox):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", array.array, typecode='f', fitness=creator.FitnessMax)
        toolbox.register("attr_float", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, self.settings['n_weights'])

    def log_helper(self):
        return WeightsLogHelper()

    def eval(self, individual):
        raise NotImplementedError()
        # return sum(individual),
