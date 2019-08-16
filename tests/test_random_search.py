from allentune.util.random_search import RandomSearch
import pytest
import numpy as np
import string


class TestRandomSearch(object):
    
    def test_random_choice(self):
        random_search = RandomSearch()
        items = range(100)
        sampler = random_search.random_choice(*items)
        for _ in range(3):
            res = sampler()
            assert res in items

    def test_random_integer(self):
        random_search = RandomSearch()
        lower_bound = np.random.choice(range(100))
        upper_bound = np.random.choice(range(100, 200))
        sampler = random_search.random_integer(lower_bound, upper_bound)
        for _ in range(3):
            res = sampler()
            assert res >= lower_bound and res <= upper_bound
    
    def test_random_loguniform(self):
        random_search = RandomSearch()
        sampler = random_search.random_loguniform(1e-5, 1e-1)
        for _ in range(3):
            res = sampler()
            assert res >= 1e-5 and res <= 1e-1

    def test_random_subset(self):
        random_search = RandomSearch()
        items = list(string.ascii_lowercase)
        sampler = random_search.random_subset(*items)
        for _ in range(3):
            res = sampler()
            assert len(res) <= len(items) and all([item in items for item in res])

    def test_random_pair(self):
        random_search = RandomSearch()
        items = list(string.ascii_lowercase)
        sampler = random_search.random_pair(*items)
        for _ in range(3):
            res = sampler()
            assert len(res) == 2 and all([item in items for item in res])

    def test_random_uniform(self):
        random_search = RandomSearch()
        sampler = random_search.random_uniform(0, 1)
        for _ in range(3):
            res = sampler()
            assert res >= 0 and res <= 1