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
import re
import warnings

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

    if np.max(obs) > n_symbols:
        raise Exception('Observation contains an invalid state:'
                        ' {}'.format(np.max(obs)))


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


def estimate_from_paths(paths, observations, n_states,
                        n_symbols, emit=False):
    """A single iteration of Viterbi training.

    Constrains transition matrix to be symmetric with a constant
    diagonal.

    """
    pseudocount = 0.1
    # count transitions to same state
    a = 0

    # unconstrained emission matrix
    E = np.zeros((n_states, n_symbols)) + pseudocount

    # count emissions matching state
    e = 0
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

    # number of unambiguous emissions
    e_denom = (obs.sum(axis=1) == 1).sum()
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
        S, A, E = estimate_from_paths(paths, observations, n_states,
                                      n_symbols, emit=emit)
        if np.allclose(S_old, S) and \
           np.allclose(A_old, A) and \
           np.allclose(E_old, E):
            break
    if i == max_iters - 1:
        warnings.warn('viterbi training forced to stop after '
                      ' {} iterations'.format(max_iters))
    return S, A, E


def preprocess(parents, child):
    """1-hot encoding for which parent each position matches.

    `result[i, j]` is True if the child matches parent `j` in position `i`.

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
    """run `preprocess()`, but then convert 1-hot encoding to binary
    encoding and mask positions where both parents match.

    Also mask terminal gaps.

    Only works for two parents.

    """
    if len(parents) != 2:
        raise Exception('map_obs() currently only works with two parents')
    obs = preprocess(parents, child)
    result = np.zeros(len(obs))
    result[obs[:, 1]] = 1
    mask = obs[:, 0] == obs[:, 1]
    start, stop = range_without_gaps(str(child.seq))
    mask[:start] = True
    mask[stop:] = True
    return np.ma.masked_array(result, mask)


def logP_single(n, N):
    """
    n: number of consistent observed symbols
    N: total observations

    """
    if n == N or n == 0:
        return 0
    p = n / N
    q = 1 - p
    return n * np.log(p) + (N - n) * np.log(q)


def range_without_gaps(cseq):
    pattern = r'[^-]'
    start = re.search(pattern, cseq).start()
    stop = len(cseq) - re.search(pattern, cseq[::-1]).start()
    return start, stop


def find_recombination(parents, child, emit=False, fast=False):
    """Run the model on a child sequence.

    Extracts relevent positions, trains a model using Viterbi
    training, uses posterior probabilities to interpolate results
    between those positions, and does a hard assignment for each
    position.

    Gaps in all three sequences are masked.

    """

    # find and remove terminal gaps in child sequence
    cseq = str(child.seq)
    start, stop = range_without_gaps(cseq)
    cseq = child.seq[start: stop]
    pseqs = list(p.seq[start: stop] for p in parents)

    observation = preprocess(pseqs, cseq)
    # re-encode all 0s as all 1s; i.e. maximally uninformative
    observation[observation.sum(axis=1) == 0, :] = 1

    # now each individual observation is either (0, 1), (1, 0), or
    # (1, 1). The idea is that when the observation is (1, 1), the
    # emission probability will be e + (1 - e) == 1, so it will not
    # contribute to the log probability at all.

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
    logprobs, logP2 = posterior_logprobs(observation, S, A, E)
    probs = np.exp(logprobs)

    n = observation[:, 0].sum()
    N = len(observation)
    logP1 = logP_single(n, N)

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
    return np.ma.masked_array(full_result, mask), logP2, logP1


def progress(xs):
    n = len(xs)
    for i, x in enumerate(xs):
        print("\rprocessing {} / {}".format(i + 1, n), end="")
        yield x
    print("")


def bic(logL, k, n):
    """Bayesian information criterion

    logL: log likelihood
    k: number of parameters
    n: sample size
    """
    return -2 * logL + k * np.log(n)


def aic(logL, k):
    return 2 * k - 2 * logL


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

    results = list(find_recombination(parents, c,
                                      emit=args['--emit'],
                                      fast=args['--fast'])
                   for c in process_children)
    logprobs, logP2s, logP1s = zip(*results)
    logprobs = np.ma.vstack(logprobs)

    outfile = args["<outfile>"]
    np.savetxt("{}.txt".format(outfile),
               logprobs.filled(-1), fmt="%.4f", delimiter=",")

    cmap = plot.cm.get_cmap('jet')
    cmap.set_bad('grey')
    plot.imsave("{}-probs.png".format(outfile), logprobs, cmap=cmap)

    cmap2 = plot.cm.get_cmap('jet', 2)
    cmap2.set_bad('grey')

    gap_array = np.array(list(list(char == "-" for char in c)
                             for c in children))
    plot.imsave("{}-gaps.png".format(outfile), gap_array,
                cmap=plot.cm.gray)

    hard_states = (logprobs > 0.5).astype(np.int)
    hard_states.mask = logprobs.mask
    plot.imsave("{}-hard.png".format(outfile), hard_states, cmap=cmap2)

    if args["--verbose"]:
        obs_children = progress(children)
    else:
        obs_children = children

    # FIXME: does not take terminal gaps into account
    all_obs = np.ma.vstack(list(map_obs(parents, c) for c in obs_children))
    plot.imsave("{}-input.png".format(outfile), all_obs, cmap=cmap2)

    # write statistics
    logP1s = np.array(logP1s)
    logP2s = np.array(logP2s)

    if args["--emit"]:
        k1 = 1
        k2 = 2
    else:
        k1 = 2
        k2 = 3

    if args["--fast"]:
        ns = np.invert(all_obs.mask).sum(axis=1)
    else:
        ns = np.invert(logprobs.mask).sum(axis=1)

    bic_1s = np.array(list(bic(logP, k1, n) for logP, n in zip(logP1s, ns)))
    bic_2s = np.array(list(bic(logP, k2, n) for logP, n in zip(logP2s, ns)))

    aic_1s = np.array(list(aic(logP, k1) for logP in logP1s))
    aic_2s = np.array(list(aic(logP, k2) for logP in logP2s))

    obs_frac = (all_obs == 0).sum(axis=1) / np.invert(all_obs.mask).sum(axis=1)
    inferred_frac = (hard_states == 0).sum(axis=1) / np.invert(hard_states.mask).sum(axis=1)

    recombined = (aic_2s < aic_1s)

    df = pd.DataFrame({
        "observed_frac0": obs_frac,
        "inferred_frac0": inferred_frac,
        "logL1": logP1s,
        "logL2": logP2s,
        "BIC1": bic_1s,
        "BIC2": bic_2s,
        "AIC1": aic_1s,
        "AIC2": aic_2s,
        "recombined": recombined,
    })
    cols = [
        "observed_frac0",
        "inferred_frac0",
        "logL1",
        "logL2",
        "BIC1",
        "BIC2",
        "AIC1",
        "AIC2",
        "recombined",
    ]
    df[cols].to_csv("{}-stats.csv".format(outfile), index=False)

    print("{} / {} ({:.2f} %) recombined".format(
        recombined.sum(), len(recombined),
        recombined.sum() / len(recombined) * 100))
