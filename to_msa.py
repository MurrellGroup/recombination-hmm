#!/usr/bin/env python

"""Combine pairwise alignments to a reference into an MSA. Discards
insertions.

Input: FASTA file containing pairwise alignments, query first.

Usage:
  to_msa.py [options] <infile> <calnfile> <seqfile> <reffile> <outfile>
  to_msa.py -h

Options:
  -r --restore-insertions  Restore terminal insertions.
  -k --keep-insertions  Stop after restoring indels. Write pairs.
  -v --verbose  Show progress
  -h --help     Show this screen

"""
import re
from collections import defaultdict
from itertools import groupby

from docopt import docopt
from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np


def ungap(s):
    return ''.join(c for c in s if c != '-')


def progress(items, verbose=False):
    items = list(items)
    total = len(items)
    for i, item in enumerate(items):
        if verbose and i % 100 == 0:
            print('\r\tprocessing {} / {}'.format(i, total), end="")
        yield item
    if verbose:
        print("")


def rle_length(part):
    return 1 if len(part) == 1 else int(part[:-1])


def terminal_indel_length(caln, mchar):
    """
    >>> terminal_indel_length("I2M", "I")
    (1, 0)

    >>> terminal_indel_length("2IM2D", "I")
    (2, 0)

    >>> terminal_indel_length("DM2D", "D")
    (1, 2)

    >>> terminal_indel_length("32M", "D")
    (0, 0)

    """
    if mchar not in ("I", "D"):
        raise Exception("unrecognized mutation character: {}".format(mchar))
    parts = re.findall(r'[0-9]*[IDM]', caln)
    first = parts[0]
    last = parts[-1]
    regexp = r'[0-9]*{}'.format(mchar)
    a = rle_length(first) if re.match(regexp, first) else 0
    b = rle_length(last) if re.match(regexp, last) else 0
    return a, b


def restore_terminal_indels(seq, ref, caln, fullseq, fullref, insertions=False):
    """

    >>> restore_terminal_indels("AAA", "AAA", "2I3M3I", "AAA", "TTAAAGGG", insertions=False)
    ('--AAA---', 'TTAAAGGG')

    >>> restore_terminal_indels("AAA", "AAA", "2D3M3I", "CCAAA", "AAAGGG", insertions=True)
    ('CCAAA---', '--AAAGGG')

    >>> restore_terminal_indels("AAA", "AAA", "2D3M3I", "CCAAA", "AAAGGG", insertions=False)
    ('AAA---', 'AAAGGG')

    """
    # for some reason usearch calls insertions deletions and vice versa
    alen_i, blen_i = terminal_indel_length(caln, "I")
    ref_head = fullref[:alen_i] if alen_i else ""
    ref_tail = fullref[-blen_i:] if blen_i else ""
    seq_head = "-" * len(ref_head)
    seq_tail = "-" * len(ref_tail)
    if insertions:
        alen_d, blen_d = terminal_indel_length(caln, "D")
        seq_head_d = fullseq[:alen_d] if alen_d else ""
        seq_tail_d = fullseq[-blen_d:] if blen_d else ""
        ref_head_d = "-" * len(seq_head_d)
        ref_tail_d = "-" * len(seq_tail_d)

        seq_head = ''.join([seq_head, seq_head_d])
        seq_tail = ''.join([seq_tail, seq_tail_d])

        ref_head = ''.join([ref_head, ref_head_d])
        ref_tail = ''.join([ref_tail, ref_tail_d])

    seq = "".join((seq_head, seq, seq_tail))
    ref = "".join((ref_head, ref, ref_tail))

    if ungap(ref) != fullref:
        raise Exception('reference does not match')
    return seq, ref


def remove_insertions(a, b):
    """
    >>> remove_insertions([0, 1, float('nan'), 2, 3, 4], "AATA-G")
    [0, 1, nan, 2, 4]

    >>> remove_insertions("AATAAG", "AATA-G")
    ['A', 'A', 'T', 'A', 'G']

    """
    return list(s for s, r in zip(a, b) if r != '-')


def new_record_seq_str(record, seqstr):
    record = record[:]
    record.seq = Seq(seqstr, alphabet=record.seq.alphabet)
    return record


if __name__ == "__main__":
    args = docopt(__doc__)
    infile = args["<infile>"]
    calnfile = args["<calnfile>"]
    seqfile = args["<seqfile>"]
    reffile = args["<reffile>"]
    outfile = args["<outfile>"]

    verbose = args['--verbose']
    restore_insertions = args['--restore-insertions']

    records = list(SeqIO.parse(infile, 'fasta'))
    seq_records = records[::2]
    ref_records = records[1::2]

    full_seq_records = list(SeqIO.parse(seqfile, 'fasta'))

    lines = open(calnfile).read().strip().split("\n")
    strands, calns = list(zip(*list(line.split('\t') for line in lines)))
    reference = str(list(SeqIO.parse(reffile, 'fasta'))[0].seq).upper()

    if verbose:
        print('checking input')
    if not (len(seq_records) == len(calns)):
        raise Exception('wrong number of calns')
    for s, r in zip(seq_records, ref_records):
        if len(s) != len(r):
            raise Exception('sequences are not aligned')
    # if not len(set(s.id for s in seq_records)) == len(seq_records):
    #     raise Exception('non-unique sequence ids')

    seqs = list(str(s.seq) for s in seq_records)
    refs = list(str(r.seq) for r in ref_records)
    pairs = list(zip(seqs, refs))
    full_seq_dict = dict((s.id, str(s.seq)) for s in full_seq_records)
    full_seqs = list(full_seq_dict[s.id] for s in seq_records)

    if verbose:
        print("restoring terminal indels")
    extended = list(restore_terminal_indels(seq, ref, caln, full_seq, reference, restore_insertions)
                    for (seq, ref), caln, full_seq in progress(zip(pairs, calns, full_seqs), verbose))

    if not all(ungap(r) == reference for (_, r) in extended):
        raise Exception('references differ')

    if args['--keep-insertions']:
        ext_seqs, ext_refs = zip(*extended)
        ss = list(new_record_seq_str(r, s) for r, s in zip(seq_records, ext_seqs))
        rs = list(new_record_seq_str(r, s) for r, s in zip(ref_records, ext_refs))
        results = list(s for e in zip(ss, rs) for s in e)
        SeqIO.write(results, outfile, 'fasta')
    else:
        if verbose:
            print('removing insertions')
        aligned = list(''.join(remove_insertions(s, r))
                       for s, r in progress(extended, verbose))

        if not all(len(a) == len(reference) for a in aligned):
            raise Exception('final alignments differ in length from reference')

        results = list(new_record_seq_str(r, s) for r, s in zip(seq_records, aligned))

        SeqIO.write(results, outfile, 'fasta')
