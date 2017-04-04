import random

import array
from opt.genetic import GeneticOptimizer
from deap import creator, base, tools, algorithms


class WeightOptimizer(GeneticOptimizer):
    def individual(self, toolbox):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", array.array, typecode='f', fitness=creator.FitnessMax)
        toolbox.register("attr_float", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, self.settings['n_weights'])

    def eval(self, individual):
        raise NotImplementedError()
        # return sum(individual),
