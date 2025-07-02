import unittest
import math
from generate_dists import generate_dirichlet_dist
import random
import numpy as np
import pytest

class TestGenerateDists(unittest.TestCase):
    def test_correct_inputs7(self):
        inputs = [[0, 3, 5], [2, 4, 6], [1]]
        alphas = [10] * 10
        np.random.seed(42)
        result = generate_dirichlet_dist(7, inputs, alphas)
        np.random.seed(42) # reset random seed
        dire_1 = np.random.dirichlet(alphas)
        dire_2 = np.random.dirichlet(alphas)
        dire_3 = np.random.dirichlet(alphas)

        expected = [
            dire_1, dire_3, dire_2, dire_1, dire_2, dire_1, dire_2
        ]

        for r, e in zip(result, expected):
            self.assertTrue(np.allclose(r, e))

    def test_missing_inputs7(self):
        inputs = [[0, 3, 5], [2, 6], [1]]
        alphas = [10] * 10
        np.random.seed(42)
        with pytest.raises(ValueError, match="Distribution is wrong: any missing values or duplicated values"):
            result = generate_dirichlet_dist(7, inputs, alphas)

    def test_duplicated_inputs7(self):
        inputs = [[0, 3, 5], [2, 4, 6], [1, 0]]
        alphas = [10] * 10
        np.random.seed(42)
        with pytest.raises(ValueError, match="Distribution is wrong: any missing values or duplicated values"):
            result = generate_dirichlet_dist(7, inputs, alphas)

