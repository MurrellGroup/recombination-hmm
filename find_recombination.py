#!/usr/bin/env python

"""Infer recombination from two parent sequences to one child.

Input: a FASTA file containing aligned sequences. The first two are
the parents, and the rest are children.

Output: Probability that each position came from the second
parent. Gaps are coded as -1.

Also creates image files.

Usage:
  find_recombination.py [options] <infile> <outfile>
  find_recombination.py -h | --help

Options:
  --fast        Use only differing sites and interpolate
  --emit        Constrain emission matrix to 1 d.o.f.
  -v --verbose  Print progress
  -h --help     Show this screen

"""

import warnings

from Bio import SeqIO
from docopt import docopt
import scipy.linalg
from scipy.misc import logsumexp
import numpy as np
from matplotlib import pyplot as plot


# TODO: transpose forward, backward, and posterior probability matrices


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
        warnings.simplefilter("ignore")
        logtrans = np.log(A)
        logstart = np.log(S)
        logemission = np.log(E)

    trellis = np.zeros((n_states, len(obs)))
    backpt = np.ones((n_states, len(obs)), np.int) * -1

    trellis[:, 0] = logstart + logsumexp(logemission[:, obs[0]], axis=1)
    for t in range(1, len(obs)):
        logprobs = trellis[:, t - 1].reshape(-1, 1) + logtrans + \
                   logsumexp(logemission[:, obs[t]], axis=1).reshape(1, -1)
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
    trellis[:, 0] = logstart + logsumexp(logemission[:, obs[0]], axis=1)
    for t in range(1, len(obs)):
        logprobs = logsumexp(trellis[:, t - 1].reshape(-1, 1) + logtrans, axis=0)
        trellis[:, t] = logprobs + logsumexp(logemission[:, obs[t]], axis=1)
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
                   logsumexp(logemission[:, obs[t + 1]], axis=1).reshape(1, -1)
        trellis[:, t] = logsumexp(logprobs, axis=1)
    return trellis


def posterior_logprobs(obs, S, A, E):
    f = forward(obs, S, A, E)
    b = backward(obs, S, A, E)
    logP = logsumexp(f[:, -1])
    assert np.allclose(logsumexp(f + b - logP, axis=0), 0.0)
    return (f + b - logP)


def posterior_decode(obs, S, A, E):
    p = posterior_logprobs(obs, S, A, E)
    return np.argmax(p, axis=1)


def get_start(A):
    """Stationary distribution of A"""
    vals, vecs = scipy.linalg.eig(A, left=True, right=False)
    idx = np.argmax(vals)
    S = vecs[:, idx]
    S /= S.sum()
    return S


def baumwelch_train(observations, S, A, E):
    """Unconstrained Baum-Welch"""
    return NotImplementedError('not updated to deal with 1 of k encoding')
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
                    val = logsumexp(list(f[source, pos] + A_log[source, sink] + \
                                         E_log[sink, obs[pos + 1]] + b[sink, pos + 1]
                                         for pos in range(len(obs) - 1)))
                    A_new[source, sink] = logsumexp([A_new[source, sink], val - logP])
            for state in range(E.shape[0]):
                for symbol in range(E.shape[1]):
                    if symbol in obs:
                        val = logsumexp(list(f[state, pos] + b[state, pos]
                                             for pos in range(len(obs)) if obs[pos] == symbol))
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


def estimate_from_paths(paths, observations, n_states, n_symbols, emit=False):
    """A single iteration of Viterbi training.

    Constrains transition matrix to be symmetric with a constant diagonal

    """
    pseudocount = 0.1
    a = 0  # count transitions to same state
    E = np.zeros((n_states, n_symbols)) + pseudocount
    e = 0  # count emissions matching state
    for path, obs in zip(paths, observations):
        if obs[0].sum() == 1:
            E[path[0], obs[0]] += 1
        if obs[0].sum() == 1 and obs[0, path[0]] == 1:
            e += 1
        for i in range(1, len(path)):
            source, sink = path[i - 1], path[i]
            if source == sink:
                a += 1
            if obs[i].sum() == 1 and obs[i, path[i]] == 1:
                e += 1
            if obs[i].sum() == 1:
                E[sink, obs[i]] += 1

    # convert to probability
    a = a / (pseudocount + sum(len(p) - 1 for p in paths))

    e_denom = (obs.sum(axis=1) == 1).sum()  # number of unambiguous emissions
    e = e / (pseudocount + e_denom)
    A = np.array([[a, 1 - a],
                  [1 - a, a]])
    if emit:
        E = np.array([[e, 1 - e],
                      [1 - e, e]])
    else:
        E = E / E.sum(axis=1).reshape((-1, 1))
    # set start probabilities to equilibrium distribution
    S = get_start(A)
    return S, A, E


def viterbi_train(observations, S, A, E, emit=False, max_iters=100):
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
        S, A, E = estimate_from_paths(paths, observations, n_states, n_symbols, emit=emit)
        if np.allclose(S_old, S) and \
           np.allclose(A_old, A) and \
           np.allclose(E_old, E):
            break
    if i == max_iters - 1:
        warnings.warn('viterbi training forced to stop after {} iterations'.format(max_iters))
    return S, A, E


def preprocess(parents, child):
    """Encode which parent each position matches.

    >>> preprocess("AATT", "tagt", "TATA")
    [[0, 1], [1, 1], [1, 0], [0, 0]]

    """
    parents = list(p.upper() for p in parents)
    child = child.upper()
    observation = []
    for i in range(len(child)):
        result = []
        for j in range(len(parents)):
            result.append(child[i] == parents[j][i])
        observation.append(result)
    return np.vstack(observation)


def map_obs(parents, child):
    if len(parents) != 2:
        raise Exception('map_obs() currently only works with two parents')
    obs = preprocess(parents, child)
    result = np.zeros(len(obs))
    result[obs[:, 1]] = 1
    mask = obs[:, 0] == obs[:, 1]
    return np.ma.masked_array(result, mask)


def L(n, N):
    p = n / N
    q = 1 - p
    return p ** n * q ** (N - n)


def find_recombination(parents, child, emit=False, fast=False):
    """Run the model on a child sequence.

    Extracts relevent positions, trains a model using Viterbi
    training, uses posterior probabilities to interpolate results
    between those positions, and does a hard assignment for each
    position.

    Gaps in all three sequences are masked.

    """
    pseqs = list(p.seq for p in parents)
    observation = preprocess(pseqs, child.seq)
    # re-encode all 0s as all 1s; i.e. maximally uninformative
    observation[observation.sum(axis=1) == 0, :] = 1

    if fast:
        # keep only differing parts
        positions = np.where(observation.sum(axis=1) == 1)[0]
        observation = observation[positions]

    S = np.array([0.5, 0.5])
    A = np.array([[0.9, 0.1],
                  [0.1, 0.9]])
    E = np.array([[0.9, 0.1],
                  [0.1, 0.9]])
    S, A, E = viterbi_train([observation], S, A, E, emit=emit)

    # FIXME: this only works for two parents
    probs = np.exp(posterior_logprobs(observation, S, A, E))

    if fast:
        # now interpolate back
        # probability that position came from parent 1
        result = np.zeros(len(child))

        # fill first part
        result[:positions[0]] = probs[1, 0]

        # interpolate
        for i in range(len(positions) - 1):
            pos1 = positions[i]
            pos2 = positions[i + 1]
            # interpolate probs[0, i] to probs[0, i + 1]
            result[pos1:pos2 + 1] = np.linspace(probs[1, i],
                                                probs[1, i + 1],
                                                pos2 - pos1 + 1)
        # fill last part
        result[positions[-1]:] = probs[1, -1]

        # # single parent model
        # likelihood = L(n, N)
    else:
        result = probs[1, :]

    # insert gaps if shared by all three
    mask = np.zeros(len(result), dtype=np.bool)
    for i in range(len(result)):
        if child[i] == '-':
            if set(p[i] for p in parents) == set('-'):
                mask[i] = True
    return np.ma.masked_array(result, mask)


def progress(xs):
    n = len(xs)
    for i, x in enumerate(xs):
        print("\rprocessing {} / {}".format(i + 1, n), end="")
        yield x


if __name__ == "__main__":
    args = docopt(__doc__)
    filename = args["<infile>"]
    reads = list(SeqIO.parse(filename, 'fasta'))
    parents = reads[:2]
    children = reads[2:]

    if args["--verbose"]:
        process_children = progress(children)
    else:
        process_children = children

    results = np.ma.vstack(list(find_recombination(parents, c,
                                                   emit=args['--emit'],
                                                   fast=args['--fast'])
                                for c in process_children))

    outfile = args["<outfile>"]
    np.savetxt("{}.txt".format(outfile), results.filled(-1), fmt="%.4f", delimiter=",")

    cmap = plot.cm.get_cmap('jet')
    cmap.set_bad('grey')
    plot.imsave("{}-probs.png".format(outfile), results, cmap=cmap)

    cmap2 = plot.cm.get_cmap('jet', 2)
    cmap2.set_bad('grey')

    hard_results = (results > 0.5).astype(np.int)
    plot.imsave("{}-hard.png".format(outfile), hard_results, cmap=cmap2)

    all_obs = np.ma.vstack(list(map_obs(parents, c) for c in children))
    plot.imsave("{}-input.png".format(outfile), all_obs, cmap=cmap2)
