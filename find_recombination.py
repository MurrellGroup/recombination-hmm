#!/usr/bin/env python

"""Infer recombination from two parent sequences to one child.

Input: a FASTA file containing aligned sequences. The first two are
the parents, and the rest are children.

Output: a text file with one line per child, in order. Each
position in that sequence is assigned a space-seperated code.

  0 = parent 0
  1 = parent 1
  - = gap in all three

Usage:
  find_recombination.py [options] <infile> <outfile>
  find_recombination.py -h | --help

Options:
  -h --help  Show this screen

"""

import warnings

from Bio import SeqIO
from docopt import docopt
import scipy.linalg
from scipy.misc import logsumexp
import numpy as np


def _check_matrices(obs, S, A, E):
    S = S.ravel()
    n_states = A.shape[0]
    n_symbols = E.shape[1]
    if A.shape != (n_states, n_states):
        raise Exception('Transmission matrix shape {} is not'
                        ' square.'.format(A.shape))
    if len(S) != n_states:
        raise Exception('Wrong number of starting'
                        ' probabilities. Expected {} and got'
                        ' {}.'.format(n_states, len(S)))
    if E.shape != (n_states, n_symbols):
        raise Exception('Emission matrix has wrong number'
                        ' of states. Expected {} and got'
                        ' {}.'.format(n_states, E.shape[0]))

    if not np.allclose(np.sum(A, axis=1), 1):
        raise Exception('Transmission probabilities do not sum to 1.')
    if not np.allclose(S.sum(), 1):
        raise Exception('Starting probabilities do not sum to 1.')
    if not np.allclose(np.sum(E, axis=1), 1):
        raise Exception('Emission probabilities do not sum to 1.')

    if np.max(obs) > n_symbols:
        raise Exception('Observation contains an invalid state:'
                        ' {}'.format(np.max(obs)))


def viterbi_decode(obs, S, A, E):
    """Viterbi algorithm.

    A: (n_states x n_states) [i, j]: transition from i to j
    E: (n_states x n_symbols)
    S: (n_states,)

    """
    _check_matrices(obs, S, A, E)
    S = S.ravel()
    n_states = A.shape[0]

    with warnings.catch_warnings():
        # already checked probabilities, so should be safe to ignoring
        # warnings
        warnings.simplefilter("ignore")
        logtrans = np.log(A)
        logstart = np.log(S)
        logemission = np.log(E)

    trellis = np.zeros((n_states, len(obs)))
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


def forward(obs, S, A, E):
    _check_matrices(obs, S, A, E)
    S = S.ravel()
    n_states = A.shape[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logtrans = np.log(A)
        logstart = np.log(S)
        logemission = np.log(E)

    trellis = np.zeros((n_states, len(obs)))
    trellis[:, 0] = logstart + logemission[:, obs[0]]
    for t in range(1, len(obs)):
        logprobs = logsumexp(trellis[:, t - 1].reshape(-1, 1) + logtrans, axis=0)
        trellis[:, t] = logprobs + logemission[:, obs[t]]
    return trellis


def backward(obs, S, A, E):
    _check_matrices(obs, S, A, E)
    S = S.ravel()
    n_states = A.shape[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logtrans = np.log(A)
        logstart = np.log(S)
        logemission = np.log(E)

    trellis = np.zeros((n_states, len(obs)))
    trellis[:, -1] = 0
    for t in reversed(range(len(obs) - 1)):
        logprobs = logtrans + \
                   trellis[:, t + 1].reshape(1, -1) + \
                   logemission[:, obs[t + 1]].reshape(1, -1)
        trellis[:, t] = logsumexp(logprobs, axis=1)
    return trellis


def posterior_logprobs(obs, S, A, E):
    f = forward(obs, S, A, E)
    b = backward(obs, S, A, E)
    logP = logsumexp(f[:, -1])
    assert np.allclose(logsumexp(f + b - logP, axis=0), 0.0)
    return f + b - logP


def posterior_decode(obs, S, A, E):
    p = posterior_logprobs(obs, S, A, E)
    return np.argmax(p, axis=0)


def get_start(A):
    """Stationary distribution of A"""
    vals, vecs = scipy.linalg.eig(A, left=True, right=False)
    idx = np.argmax(vals)
    S = vecs[:, idx]
    S /= S.sum()
    return S


def baumwelch_train(observations, S, A, E):
    """Unconstrained Baum-Welch"""
    while True:
        # implicit pseudocount of 1
        A_new = np.zeros_like(A)
        E_new = np.zeros_like(E)
        S_log = np.log(S)
        A_log = np.log(A)
        E_log = np.log(E)
        for obs in observations:
            f = forward(obs, S, A, E)
            b = backward(obs, S, A, E)
            logP = logsumexp(f[:, -1])
            for source in range(A.shape[0]):
                for sink in range(A.shape[1]):
                    val = logsumexp(list(f[source, pos] + A_log[source, sink] + E_log[sink, obs[pos + 1]] + b[sink, pos + 1]
                                         for pos in range(len(obs) - 1)))
                    A_new[source, sink] = logsumexp([A_new[source, sink], val - logP])
            for state in range(E.shape[0]):
                for symbol in range(E.shape[1]):
                    if symbol in obs:
                        val = logsumexp(list(f[state, pos] + b[state, pos] for pos in range(len(obs)) if obs[pos] == symbol))
                        E_new[state, symbol] = logsumexp([E_new[state, symbol], val - logP])
        A_new = np.exp(A_new)
        E_new = np.exp(A_new)
        A_new = A_new / A_new.sum(axis=1).reshape((-1, 1))
        E_new = E_new / E_new.sum(axis=1).reshape((-1, 1))
        S = get_start(A_new)
        if np.allclose(A_new, A) and np.allclose(E_new, E):
            break
        A = A_new
        E = E_new
    return S, A_new, E_new


def estimate_from_paths(paths, observations, n_states, n_symbols):
    """A single iteration of Viterbi training.

    Constrains transition matrix to be symmetric with a constant diagonal

    """
    pseudocount = 0.1
    a = 0  # count transitions to same state
    E = np.zeros((n_states, n_symbols)) + pseudocount
    for path, obs in zip(paths, observations):
        E[path[0], obs[0]] += 1
        for i in range(1, len(path)):
            source, sink = path[i - 1], path[i]
            if source == sink:
                a += 1
            E[sink, obs[i]] += 1
    # convert to probability
    a = a / (pseudocount + sum(len(p) - 1 for p in paths))
    A = np.array([[a, 1 - a],
                  [1 - a, a]])
    E = E / E.sum(axis=1).reshape((-1, 1))
    # set start probabilities to equilibrium distribution
    S = get_start(A)
    return S, A, E


def viterbi_train(observations, S, A, E, max_iters=100):
    n_states = A.shape[0]
    n_symbols = E.shape[1]
    assert A.shape == (n_states, n_states)
    assert E.shape == (n_states, n_symbols)
    for i in range(max_iters):
        S_old = S.copy()
        A_old = A.copy()
        E_old = E.copy()
        paths = list(viterbi_decode(o, S, A, E)
                     for o in observations)
        S, A, E = estimate_from_paths(paths,
                                      observations,
                                      n_states,
                                      n_symbols)
        if np.allclose(S_old, S) and \
           np.allclose(A_old, A) and \
           np.allclose(E_old, E):
            break
    return S, A, E


def preprocess(p1, p2, child):
    """Keep child positions that match exactly one parent.

    Returns:
      * observation: array of 0s (for parent 1) and 1s (for parent 2)
      * positions: indices of the observations in the full sequence

    >>> preprocess("AAT", "TAG", "TAT")
    [1, 0], [0, 2]

    """
    observation = []
    positions = []
    for i, (a, b, c) in enumerate(zip(p1.seq, p2.seq, child.seq)):
        if ((c == a) or (c == b)) and (a != b):
            observation.append(0 if c == a else 1)
            positions.append(i)
    return observation, positions


def find_recombination(p1, p2, child):
    """Run the model on a child sequence.

    Extracts relevent positions, trains a model using Viterbi
    training, uses posterior probabilities to interpolate results
    between those positions, and does a hard assignment for each
    position.

    Gaps in all three sequences are coded as -1.

    """
    observation, positions = preprocess(p1, p2, child)
    S = np.array([0.5, 0.5])
    A = np.array([[0.9, 0.1],
                  [0.1, 0.9]])
    E = np.array([[0.9, 0.1],
                  [0.1, 0.9]])
    S, A, E = viterbi_train([observation],
                            S,
                            A,
                            E)
    probs = np.exp(posterior_logprobs(observation, S, A, E))

    # probability that position came from parent 1
    result = np.zeros(len(child))

    # fill first part
    result[:positions[0]] = probs[0, 0]

    # interpolate
    for i in range(len(positions) - 1):
        pos1 = positions[i]
        pos2 = positions[i + 1]
        # interpolate probs[0, i] to probs[0, i + 1]
        result[pos1:pos2 + 1] = np.linspace(probs[0, i],
                                             probs[0, i + 1],
                                             pos2 - pos1 + 1)
    # fill last part
    result[positions[-1]:] = probs[0, -1]

    # hard assignment
    result = (result > 0.5).astype(np.int)

    # insert gaps if shared by all three
    for i, (a, b, c) in enumerate(zip(p1, p2, child)):
        if a == b == c == "-":
            result[i] = -1

    return result


if __name__ == "__main__":
    args = docopt(__doc__)
    filename = args["<infile>"]
    reads = list(SeqIO.parse(filename, 'fasta'))
    p1 = reads[0]
    p2 = reads[1]
    children = reads[2:]

    results = list(find_recombination(p1, p2, c) for c in children)
    outfile = args["<outfile>"]
    with open(outfile, 'w') as h:
        for r in results:
            h.write(' '.join(map(lambda s: '-' if s < 0 else str(s), r)))
            h.write('\n')
