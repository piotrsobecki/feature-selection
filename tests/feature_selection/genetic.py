import unittest
from io import StringIO

import pandas as pd
from sklearn.neighbors.classification import KNeighborsClassifier

from opt.feature_selection.genetic import CVGeneticFeatureSelection


class TestStringMethods(unittest.TestCase):
    def test_foo(self):
        features = pd.DataFrame.from_dict({
            "A": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "B": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            "C": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        })
        labels = pd.DataFrame.from_dict({
            "Test": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        })

        cls = [KNeighborsClassifier()]

        fs = CVGeneticFeatureSelection(cls, features, labels, verbose=False, genlog='genlog.csv',  datalog='datalog.csv', sep=';')
        results = fs.fit()

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
