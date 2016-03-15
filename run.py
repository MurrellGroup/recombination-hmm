#!/usr/bin/env python

"""
Usage:
  run.py [options] <infile>
  run.py -h | --help

Options:
  -v --verbose            Print progress to STDERR
  -h --help               Show this screen

"""

import warnings

from Bio import SeqIO
from docopt import docopt
import scipy.linalg
from scipy.misc import logsumexp
import numpy as np


def _check_matrices(obs, startprob, transmat, emissionprob):
    startprob = startprob.ravel()
    n_states = transmat.shape[0]
    n_symbols = emissionprob.shape[1]
    if transmat.shape != (n_states, n_states):
        raise Exception('Transmission matrix shape {} is not'
                        ' square.'.format(transmat.shape))
    if len(startprob) != n_states:
        raise Exception('Wrong number of starting'
                        ' probabilities. Expected {} and got'
                        ' {}.'.format(n_states, len(startprob)))
    if emissionprob.shape != (n_states, n_symbols):
        raise Exception('Emission matrix has wrong number'
                        ' of states. Expected {} and got'
                        ' {}.'.format(n_states, emissionprob.shape[0]))

    if not np.allclose(np.sum(transmat, axis=1), 1):
        raise Exception('Transmission probabilities do not sum to 1.')
    if not np.allclose(startprob.sum(), 1):
        raise Exception('Starting probabilities do not sum to 1.')
    if not np.allclose(np.sum(emissionprob, axis=1), 1):
        raise Exception('Emission probabilities do not sum to 1.')

    if np.max(obs) > n_symbols:
        raise Exception('Observation contains an invalid state:'
                        ' {}'.format(np.max(obs)))



# copied from flea_pipeline/trim.py
def viterbi_decode(obs, startprob, transmat, emissionprob):
    """Viterbi algorithm.

    transmat: (n_states x n_states) [i, j]: transition from i to j
    emissionprob: (n_states x n_symbols)
    startprob: (n_states,)

    """
    _check_matrices(obs, startprob, transmat, emissionprob)
    startprob = startprob.ravel()
    n_states = transmat.shape[0]

    with warnings.catch_warnings():
        # already checked probabilities, so should be safe to ignoring
        # warnings
        warnings.simplefilter("ignore")
        logtrans = np.log(transmat)
        logstart = np.log(startprob)
        logemission = np.log(emissionprob)

    # heavily adapted from:
    # http://phvu.net/2013/12/06/sweet-implementation-of-viterbi-in-python/

    # dynamic programming matrix
    trellis = np.zeros((n_states, len(obs)))
    # back pointers
    backpt = np.ones((n_states, len(obs)), np.int) * -1

    trellis[:, 0] = logstart + logemission[:, obs[0]]
    for t in range(1, len(obs)):
        logprobs = trellis[:, t - 1].reshape(-1, 1) + logtrans + \
                   logemission[:, obs[t]].reshape(1, -1)
        trellis[:, t] = logprobs.max(axis=0)
        backpt[:, t] = logprobs.argmax(axis=0)
    result = [trellis[:, -1].argmax()]
    for i in range(len(obs)-1, 0, -1):
        result.append(backpt[result[-1], i])
    result = list(reversed(result))
    return result


def forward(obs, startprob, transmat, emissionprob):
    _check_matrices(obs, startprob, transmat, emissionprob)
    startprob = startprob.ravel()
    n_states = transmat.shape[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logtrans = np.log(transmat)
        logstart = np.log(startprob)
        logemission = np.log(emissionprob)

    trellis = np.zeros((n_states, len(obs)))
    trellis[:, 0] = logstart + logemission[:, obs[0]]
    for t in range(1, len(obs)):
        logprobs = logsumexp(trellis[:, t - 1].reshape(-1, 1) + logtrans, axis=0)
        trellis[:, t] = logprobs + logemission[:, obs[t]]
    return trellis


def backward(obs, startprob, transmat, emissionprob):
    _check_matrices(obs, startprob, transmat, emissionprob)
    startprob = startprob.ravel()
    n_states = transmat.shape[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logtrans = np.log(transmat)
        logstart = np.log(startprob)
        logemission = np.log(emissionprob)

    trellis = np.zeros((n_states, len(obs)))
    trellis[:, -1] = 0
    for t in reversed(range(len(obs) - 1)):
        logprobs = logtrans + \
                   trellis[:, t + 1].reshape(1, -1) + \
                   logemission[:, obs[t + 1]].reshape(1, -1)
        trellis[:, t] = logsumexp(logprobs, axis=1)
    return trellis


def posterior_logprobs(obs, startprob, transmat, emissionprob):
    f = forward(obs, startprob, transmat, emissionprob)
    b = backward(obs, startprob, transmat, emissionprob)
    logP = logsumexp(f[:, -1])
    assert np.allclose(logsumexp(f + b - logP, axis=0), 0.0)
    return f + b - logP


def posterior_decode(obs, startprob, transmat, emissionprob):
    p = posterior_logprobs(obs, startprob, transmat, emissionprob)
    return np.argmax(p, axis=0)


def estimate_from_paths(paths, observations, n_states, n_symbols):
    transmat = np.zeros((n_states, n_states)) + 1  # pseudocount
    emissionprob = np.zeros((n_states, n_symbols)) + 1  # pseudocount
    for path, obs in zip(paths, observations):
        for i in range(1, len(path)):
            source, sink = path[i - 1], path[i]
            transmat[source, sink] += 1
            emissionprob[sink, obs[i]] += 1
    # normalize transmat and emissionprob
    transmat = transmat / transmat.sum(axis=1).reshape((-1, 1))
    emissionprob = emissionprob / emissionprob.sum(axis=1).reshape((-1, 1))
    # set start probabilities to equilibrium distribution
    vals, vecs = scipy.linalg.eig(transmat, left=True, right=False)
    idx = np.argmax(vals)
    startprob = vecs[:, idx]
    startprob /= startprob.sum()
    return startprob, transmat, emissionprob


def estimate_from_expectations():
    pass


def baumwelch_train():
    pass


def viterbi_train(observations, startprob, transmat, emissionprob, max_iters=100):
    n_states = transmat.shape[0]
    n_symbols = emissionprob.shape[1]
    assert transmat.shape == (n_states, n_states)
    assert emissionprob.shape == (n_states, n_symbols)
    for i in range(max_iters):
        old_startprob = startprob.copy()
        old_transmat = transmat.copy()
        old_emissionprob = emissionprob.copy()
        paths = list(viterbi_decode(o, startprob, transmat, emissionprob)
                     for o in observations)
        startprob, transmat, emissionprob = estimate_from_paths(paths,
                                                                observations,
                                                                n_states,
                                                                n_symbols)
        if np.allclose(old_startprob, startprob) and \
           np.allclose(old_transmat, transmat) and \
           np.allclose(old_emissionprob, emissionprob):
            break
    return startprob, transmat, emissionprob


def preprocess(p1, p2, child):
    """

    >>> preprocess("AAT", "TAG", "TAT")
    [1, 0]

    """
    observation = []
    for a, b, c in zip(p1.seq, p2.seq, child.seq):
        if ((c == a) or (c == b)) and (a != b):
            observation.append(0 if c == a else 1)
    return observation


def run(observation):
    startprob = np.array([0.5, 0.5])
    transmat = np.array([[0.9, 0.1],
                         [0.1, 0.9]])
    emissionprob = np.array([[0.9, 0.1],
                             [0.1, 0.9]])
    startprob, transmat, emissionprob = viterbi_train([observation],
                                                      startprob,
                                                      transmat,
                                                      emissionprob)
    path = posterior_decode(observation, startprob, transmat, emissionprob)
    return path, startprob, transmat, emissionprob


if __name__ == "__main__":
    args = docopt(__doc__)
    filename = args["<infile>"]
    reads = SeqIO.parse(filename, 'fasta')
    p1 = reads[0]
    p2 = reads[1]
    children = reads[2:]
