import unittest
from itertools import product

import numpy as np
from numpy.testing import assert_allclose
from scipy.misc import logsumexp

from find_recombination import estimate_from_paths
from find_recombination import posterior_logprobs
from find_recombination import forward, backward
from find_recombination import precompute_emission


class TestFindRecombination(unittest.TestCase):

    def setUp(self):
        self.obs = np.array([[True, False],
                             [True, True],
                             [False, True]])
        self.S = np.array([0.5, 0.5])
        self.A = np.array([[0.9, 0.1],
                           [0.1, 0.9]])
        self.E = np.array([[0.9, 0.1],
                           [0.1, 0.9]])


    def test_estimate_from_paths(self):
        paths = [[0, 0, 0, 1, 1, 1]]
        observations = [np.array([[True, False],
                                  [True, False],
                                  [True, False],
                                  [True, False],
                                  [False, True],
                                  [False, True]])]
        n_states = 2
        n_symbols = 2
        S, A, E = estimate_from_paths(paths, observations, n_states,
                                      n_symbols, constrain=False,
                                      pseudocount=0)

        assert_allclose(S, [0.5, 0.5])
        assert_allclose(A, [[4/5, 1/5],
                            [1/5, 4/5]])
        assert_allclose(E, [[1, 0],
                            [1/3, 2/3]])

        S, A, E = estimate_from_paths(paths, observations, n_states,
                                      n_symbols, constrain=True,
                                      pseudocount=0)

        assert_allclose(S, [0.5, 0.5])
        assert_allclose(A, [[4/5, 1/5],
                            [1/5, 4/5]])
        assert_allclose(E, [[5/6, 1/6],
                            [1/6, 5/6]])

    def test_forward(self):
        f = forward(self.obs, self.S, self.A, self.E)
        exp = np.log([[0.45, 0.05],
                      [0.4055, .09],
                      [0.040995, .076955]]).T
        assert_allclose(f, exp, rtol=0.1, atol=0.1)

    def test_backward(self):
        b = backward(self.obs, self.S, self.A, self.E)
        exp = np.log([[0.244, 0.756],
                      [0.18, 0.82],
                      [1, 1]]).T
        assert_allclose(b, exp, rtol=0.1, atol=0.1)

    def test_forward_backward(self):
        """Check that forward and backward algorithms give same log probability"""
        f = forward(self.obs, self.S, self.A, self.E)
        b = backward(self.obs, self.S, self.A, self.E)
        fp = logsumexp(f[:, -1])
        emission = precompute_emission(np.log(self.E))[tuple(self.obs[0])]
        bp = logsumexp(np.log(self.S) + emission + b[:, 0])
        assert_allclose(fp, bp)

    def test_posterior_logprobs(self):
        """Check that marginalizing over all possible paths gives probability of 1"""
        x = list(product([True, False], repeat=2))
        xs = list(e for e in product(x, repeat=3))
        all_obs = list(o for o in xs
                       if all(any(e) and not all(e) for e in o))
        total = logsumexp(list(posterior_logprobs(np.array(obs), self.S, self.A, self.E)[1]
                               for obs in all_obs))
        assert_allclose(total, np.log(1))

if __name__ == '__main__':
    unittest.main()
