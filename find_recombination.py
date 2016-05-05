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
  --slow         Use all sites.
  --constrain    Constrain emission matrix to 1 d.o.f.
  --use-gaps     Use gaps as informative.
  -v --verbose   Print progress
  -h --help      Show this screen

"""
import re
import warnings
from itertools import groupby

from Bio import SeqIO
from docopt import docopt
import scipy.linalg
from scipy.misc import logsumexp
import numpy as np
from matplotlib import pyplot as plot
import pandas as pd


def _check_matrices(obs, S, A, E):
    """Ensure matrices have the correct shapes and contents.

    obs: series of observed symbols

    S: initial probabilities. shape: (n_states,)
    A: transmission matrix. shape: (n_states, n_states)
    E: emission matrix. shape: (n_states, n_symbols)

    """
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

    if obs.shape[1] != n_symbols:
        raise Exception('Observation contains wrong number of states')
    if obs.dtype != np.bool:
        raise Exception('Observation is not a boolean array')


def precompute_emission(logemission):
    cache = {}
    for key in ((False, True), (True, False)):
        cache[key] = logsumexp(logemission[:, np.array(key)], axis=1)
    # should be exactly 0, but not always is because of floating point error
    cache[(True, True)] = np.array([0, 0])
    return cache


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

    emission = precompute_emission(logemission)
    trellis = np.zeros((n_states, len(obs)))
    backpt = np.ones((n_states, len(obs)), np.int) * -1

    trellis[:, 0] = logstart + emission[tuple(obs[0])]
    for t in range(1, len(obs)):
        logprobs = trellis[:, t - 1].reshape(-1, 1) + logtrans + \
                   emission[tuple(obs[t])]
        trellis[:, t] = logprobs.max(axis=0)
        backpt[:, t] = logprobs.argmax(axis=0)
    result = [trellis[:, -1].argmax()]
    for i in range(len(obs)-1, 0, -1):
        result.append(backpt[result[-1], i])
    result = list(reversed(result))
    return result


def forward(obs, S, A, E):
    """Computes forward DP matrix.

    `trellis[i, j]` is probability of being in state `i` in position `j`
    given `obs[:j+1]`.

    """
    _check_matrices(obs, S, A, E)
    S = S.ravel()
    n_states = A.shape[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logtrans = np.log(A)
        logstart = np.log(S)
        logemission = np.log(E)

    emission = precompute_emission(logemission)

    trellis = np.zeros((n_states, len(obs)))
    trellis[:, 0] = logstart + emission[tuple(obs[0])]
    for t in range(1, len(obs)):
        logprobs = logsumexp(trellis[:, t - 1].reshape(-1, 1) + logtrans, axis=0)
        trellis[:, t] = logprobs + emission[tuple(obs[t])]
    return trellis


def backward(obs, S, A, E):
    """Computes backward DP matrix.

    trellis[i, j] is probability of being in state i in position j
    given obs[j:].

    """
    _check_matrices(obs, S, A, E)
    S = S.ravel()
    n_states = A.shape[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logtrans = np.log(A)
        logstart = np.log(S)
        logemission = np.log(E)

    emission = precompute_emission(logemission)

    trellis = np.zeros((n_states, len(obs)))
    trellis[:, -1] = 0
    for t in reversed(range(len(obs) - 1)):
        logprobs = logtrans + \
                   trellis[:, t + 1].reshape(1, -1) + \
                   emission[tuple(obs[t + 1])]
        trellis[:, t] = logsumexp(logprobs, axis=1)
    return trellis


def posterior_logprobs(obs, S, A, E):
    """Compute the posterior log probabilities for each state in each
    position, and the overall log probability.

    """
    f = forward(obs, S, A, E)
    b = backward(obs, S, A, E)
    logP = logsumexp(f[:, -1])
    assert np.allclose(logsumexp(f + b - logP, axis=0), 0.0)
    return (f + b - logP), logP


def get_start(A):
    """Stationary distribution of transmission matrix."""
    vals, vecs = scipy.linalg.eig(A, left=True, right=False)
    idx = np.argmax(vals)
    S = vecs[:, idx]
    S /= S.sum()
    return S


def estimate_from_paths(paths, observations, n_states,
                        n_symbols, constrain=False, pseudocount=0.1):
    """A single iteration of Viterbi training.

    Constrains transition matrix to be symmetric with a constant
    diagonal.

    """
    # count transitions to same state
    a = 0

    # unconstrained emission matrix
    E = np.zeros((n_states, n_symbols)) + pseudocount

    # count emissions matching state
    e = 0
    for path, obs in zip(paths, observations):
        assert obs.dtype == np.bool
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

    # number of unambiguous emissions
    e_denom = (obs.sum(axis=1) == 1).sum()
    e = e / (pseudocount + e_denom)
    A = np.array([[a, 1 - a],
                  [1 - a, a]])
    if constrain:
        E = np.array([[e, 1 - e],
                      [1 - e, e]])
    else:
        E = E / E.sum(axis=1).reshape((-1, 1))
    # set start probabilities to equilibrium distribution
    S = get_start(A)
    return S, A, E


def viterbi_train(observations, S, A, E, constrain=False, max_iters=100):
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
        S, A, E = estimate_from_paths(paths, observations, n_states,
                                      n_symbols, constrain=constrain)
        if np.allclose(S_old, S) and \
           np.allclose(A_old, A) and \
           np.allclose(E_old, E):
            break
    if i == max_iters - 1:
        warnings.warn('viterbi training forced to stop after '
                      ' {} iterations'.format(max_iters))
    return S, A, E


def preprocess(parents, child, ignore_gaps=True):
    """1-hot encoding for which parent each position matches.

    `result[i, j]` is True if the child matches parent `j` in position `i`.

    >>> list(preprocess(['AAA', 'AAC'], 'ATA', ignore_gaps=False).ravel())
    [True, True, True, True, True, False]

    >>> list(preprocess(['A-A', 'AAC'], 'A-A', ignore_gaps=True).ravel())
    [True, True, True, True, True, False]

    """
    aln_len = len(child)
    if not all(len(p) == aln_len for p in parents):
        raise Exception('parents and child are not aligned')
    parents = list(p.upper() for p in parents)
    child = child.upper()
    observation = []
    for i in range(len(child)):
        result = []
        if ignore_gaps and (child[i] == '-' or any(p[i] == '-' for p in parents)):
            result.append([False] * len(parents))
        else:
            for j in range(len(parents)):
                result.append(child[i] == parents[j][i])
        observation.append(np.array(result, dtype=np.bool))
    result = np.vstack(observation)
    # re-encode all 0s as all 1s; i.e. maximally uninformative
    result[~result[:, 0] & ~result[:, 1]] = True
    return result


def map_obs(parents, child, ignore_gaps=True):
    """run `preprocess()`, but then convert 1-hot encoding to binary
    encoding and mask positions where both parents match.

    Also mask terminal gaps.

    Only works for two parents.

    >>> list(map_obs(['A-AA', 'AACT'], 'A-AT', ignore_gaps=True).filled(-1))
    [-1, -1, 0, 1]

    """
    if len(parents) != 2:
        raise Exception('map_obs() currently only works with two parents')
    obs = preprocess(parents, child, ignore_gaps=ignore_gaps)
    result = np.zeros(len(obs), dtype=np.int)
    positions = np.array(~obs[:, 0] & obs[:, 1])
    result[positions] = True
    mask = (obs[:, 0] == obs[:, 1])
    start, stop = range_without_gaps(child)
    mask[:start] = True
    mask[stop:] = True
    final = np.ma.masked_array(result, mask)
    return final


def logP_single(observation):
    """Log prob of single parent model.

    Only considers (0, 1) or (1, 0) positions.

    >>> logP_single(np.array([[0, 1], [0, 1]]))
    0

    >>> logP_single(np.array([[0, 1], [1, 0]]))
    -1.3862943611198906

    """
    n = 0
    N = 0
    for i in range(len(observation)):
        if np.all(observation[i] == np.array([1, 0])):
            n += 1
            N += 1
        elif np.all(observation[i] == np.array([0, 1])):
            N += 1
    if n == N or n == 0:
        return 0
    p = n / N
    q = 1 - p
    return n * np.log(p) + (N - n) * np.log(q)


def range_without_gaps(cseq):
    """

    >>> range_without_gaps('---ACGTT-')
    (3, 8)

    """
    pattern = r'[^-]'
    start = re.search(pattern, cseq).start()
    stop = len(cseq) - re.search(pattern, cseq[::-1]).start()
    return start, stop


def find_recombination(parents, child, constrain=False, fast=True, ignore_gaps=True):
    """Run the model on a child sequence.

    Extracts relevent positions, trains a model using Viterbi
    training, uses posterior probabilities to interpolate results
    between those positions, and does a hard assignment for each
    position.

    Gaps in all three sequences are masked.

    """
    # find and remove terminal gaps in child sequence
    start, stop = range_without_gaps(child)
    cseq = child[start: stop]
    pseqs = list(p[start: stop] for p in parents)

    observation = preprocess(pseqs, cseq, ignore_gaps=ignore_gaps)

    # now each individual observation is either (0, 1), (1, 0), or
    # (1, 1). The idea is that when the observation is (1, 1), the
    # emission probability will be e + (1 - e) == 1, so it will not
    # contribute to the log probability at all.

    if fast:
        positions = np.where(observation.sum(axis=1) == 1)[0]
        observation = observation[positions]

    S = np.array([0.5, 0.5])
    A = np.array([[0.9, 0.1],
                  [0.1, 0.9]])
    E = np.array([[0.9, 0.1],
                  [0.1, 0.9]])
    S, A, E = viterbi_train([observation], S, A, E, constrain=constrain)

    # FIXME: this only works for two parents
    logprobs, logP2 = posterior_logprobs(observation, S, A, E)
    probs = np.exp(logprobs)

    logP1 = logP_single(observation)

    if fast:
        # now interpolate back
        # probability that position came from parent 1
        result = np.zeros(len(cseq))
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

    else:
        result = probs[1, :]

    # map back to full alignment coordinates
    full_result = np.zeros(len(child))
    full_result[start:stop] = result

    # mask terminal gaps, and gaps that are shared by all three
    mask = np.zeros(len(full_result), dtype=np.bool)
    for i in range(len(full_result)):
        if i < start:
            mask[i] = True
        if i >= stop:
            mask[i] = True
        if child[i] == '-':
            if set(p[i] for p in parents) == set('-'):
                mask[i] = True

    final = np.ma.masked_array(full_result, mask)
    return final, logP2, logP1


def progress(xs, verbose=False):
    n = len(xs)
    for i, (k, v) in enumerate(xs.items()):
        if verbose:
            print("\rprocessing {} / {}".format(i + 1, n), end='')
        yield k, v
    if verbose:
        print("")


def aic(logL, k):
    return 2 * k - 2 * logL


def bic(logL, k, n):
    return - 2 * logL + k * np.log(n)


if __name__ == "__main__":
    args = docopt(__doc__)
    filename = args["<infile>"]
    outfile = args["<outfile>"]
    verbose = args["--verbose"]
    ignore_gaps = not args['--use-gaps']

    records = list(SeqIO.parse(filename, 'fasta'))
    children = records[::3]
    parent_0s = records[1::3]
    parent_1s = records[2::3]

    cdict = {}
    for c, p0, p1 in zip(children, parent_0s, parent_1s):
        key = str(c.seq)
        if key not in cdict:
            cdict[key] = (str(p0.seq), str(p1.seq))

    if verbose:
        print('reduced {} to {} unique'.format(len(children), len(cdict)))

    if verbose:
        print('computing observations')
    all_obs_dict = dict((child, map_obs(parents, child, ignore_gaps=ignore_gaps))
                        for child, parents in progress(cdict, verbose))
    all_obs = list(all_obs_dict[str(c.seq)] for c in children)

    if verbose:
        print('finding recombination')
    results_dict = dict((child, find_recombination(parents, child,
                                               constrain=args['--constrain'],
                                               fast=not args['--slow'],
                                               ignore_gaps=ignore_gaps))
                        for child, parents in progress(cdict, verbose))
    results = list(results_dict[str(c.seq)] for c in children)
    logprobs, logP2s, logP1s = zip(*results)

    # write statistics
    logP1s = np.array(logP1s)
    logP2s = np.array(logP2s)

    k1 = 1
    k2 = 2 if args['--constrain'] else 3

    aic_1s = np.array(list(aic(logP, k1) for logP in logP1s))
    aic_2s = np.array(list(aic(logP, k2) for logP in logP2s))
    aic_mins = np.vstack([aic_1s, aic_2s]).min(axis=0)
    rel_probs1 = np.exp((aic_mins - aic_1s) / 2)
    rel_probs2 = np.exp((aic_mins - aic_2s) / 2)

    n_informative = list(np.invert(o.mask).sum() for o in all_obs)
    ns = n_informative
    bic_1s = np.array(list(bic(logP, k1, n) for logP, n in zip(logP1s, ns)))
    bic_2s = np.array(list(bic(logP, k2, n) for logP, n in zip(logP2s, ns)))

    hard_states = []
    for ps in logprobs:
        hard = (ps > 0.5).astype(np.int)
        hard.mask = ps.mask
        hard_states.append(hard)

    df = pd.DataFrame({
        'label': list(c.id for c in children),
        'n_positions': list(len(c) for c in children),
        'n_informative': n_informative,
        "n_informative_0": list((o == 0).sum() for o in all_obs),
        "n_informative_1": list((o == 1).sum() for o in all_obs),
        "n_inferred": list(np.invert(h.mask).sum() for h in hard_states),
        "n_inferred_0": list((h == 0).sum() for h in hard_states),
        "n_inferred_1": list((h == 1).sum() for h in hard_states),
        "k1": k1,
        "k2": k2,
        "logL1": logP1s,
        "logL2": logP2s,
        "BIC1": bic_1s,
        "BIC2": bic_2s,
        "AIC1": aic_1s,
        "AIC2": aic_2s,
        "rel_prob1": rel_probs1,
        "rel_prob2": rel_probs2,
    })
    cols = [
        'label',
        'n_positions',
        'n_informative',
        "n_informative_0",
        "n_informative_1",
        "n_inferred",
        "n_inferred_0",
        "n_inferred_1",
        "k1",
        "k2",
        "logL1",
        "logL2",
        "BIC1",
        "BIC2",
        "AIC1",
        "AIC2",
        "rel_prob1",
        "rel_prob2",
    ]
    df[cols].to_csv("{}-stats.csv".format(outfile), index=False)

    recombined = rel_probs2 > rel_probs1
    print("{} / {} ({:.2f} %) recombined".format(
        recombined.sum(), len(recombined),
        recombined.sum() / len(recombined) * 100))
