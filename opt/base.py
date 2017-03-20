import json


class Configuration:
    def __init__(self, individual):
        self.individual = individual

    def __str__(self):
        return "(%.4f): %s" % (self.value(), self.individual)

    def value(self):
        raise NotImplementedError()


class Results:
    def __init__(self, population):
        self.population = population

    def value_accessor(self, configuration):
        return configuration.value()

    def fitness(self):
        return map(self.value_accessor, self.population)

    def max(self):
        return max(self.population, key=self.value_accessor)

    def min(self):
        return min(self.population, key=self.value_accessor)

    def __str__(self):
        return json.dumps(self.__dict__, indent=4, sort_keys=True)


class Optimizer():
    def fit(self):
        pass
