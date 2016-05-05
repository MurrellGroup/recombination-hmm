import sys
import re
from collections import defaultdict

from Bio import SeqIO

from to_msa import ungap, remove_insertions, new_record_seq_str


def ungapped_seq_range(seq, parent):
    """Get ungapped sequence range that corresponds to trimming
    insertions relative to the parent.

    >>> ungapped_seq_range('-AC-T-', 'ATCGTT')
    (0, 3)

    >>> ungapped_seq_range('AAAGGGAAA', '---GGG---')
    (3, 6)

    """

    aln_start = re.search('[^-]', parent).start()
    aln_stop = len(parent) - re.search('[^-]', parent[::-1]).start()

    seq_len = sum(1 for c in seq if c != '-')
    head_len = sum(1 for c in seq[:aln_start] if c != '-')
    tail_len = sum(1 for c in seq[aln_stop:] if c != '-')

    return head_len, seq_len - tail_len


def trim_alignment(seq, parent, start, stop):
    """

    >>> trim_alignment('-ACGTAAA', 'TACTT---', 0, 4)
    ('ACGT', 'ACTT')

    """
    assert start <= stop
    seq_idx = -1
    a = -1
    b = -1
    for i, c in enumerate(seq):
        if c != '-':
            seq_idx += 1
        if seq_idx == start:
            a = i
        if seq_idx == stop - 1:
            b = i + 1
    if a == -1 or b == -1:
        assert False
    return seq[a:b], parent[a:b]


def trim_to_parents(pairs):
    """Trim alignments.

    Pairs are (seq, parent) alignments.

    >>> trim_to_parents([['TTAAAT', '--AAA-'], ['TTAAAT', '-TAAA-']])
    [('TAAA', '-AAA'), ('TAAA', 'TAAA')]

    >>> trim_to_parents([['-AC-T', 'TTTTT'], ['ACT', 'GGG']])
    [('AC-T', 'TTTT'), ('ACT', 'GGG')]

    """
    # get ungapped sequence range
    ranges = list(ungapped_seq_range(s, p) for s, p in pairs)
    starts, stops = list(zip(*ranges))
    start = min(starts)
    stop = max(stops)

    # trim each alignment to match that range
    result = list(trim_alignment(s, p, start, stop) for s, p in pairs)

    result_seq = ungap(result[0][0])
    assert all(ungap(s) == result_seq for s, _ in result)
    return result


if __name__ == "__main__":
    infile, outfile = sys.argv[1:]

    records = list(SeqIO.parse(infile, 'fasta'))

    seqdict = defaultdict(list)
    for seq, parent in zip(records[::2], records[1::2]):
        seqdict[seq.id].append((seq, parent))

    n_parents = max(len(v) for v in seqdict.values())
    missing_parents = list(k for k, v in seqdict.iteritems() if len(v) < n_parents)
    if missing_parents:
        raise Exception('some sequences are missing some parents')

    result = []
    for label, pairs in seqdict.items():
        if len(set(p.id for s, p in pairs)) < len(pairs):
            raise Exception('non-unique parents')
        seq = ungap(pairs[0][0])
        if not all(ungap(s) == seq for s, _ in pairs):
            raise Exception('seqs for {} are not equal'.format(label))

        pair_seqs = list((str(s.seq), str(p.seq)) for s, p in pairs)

        # trim to parents
        trimmed = trim_to_parents(pair_seqs)
        seq = ungap(trimmed[0][0])

        # remove point insertions
        aligned_parents = list(''.join(remove_insertions(p, s)) for s, p in trimmed)
        result.append(new_record_seq_str(pairs[0][0], seq))
        result.extend(list(new_record_seq_str(p, new_seq)
                           for (s, p), new_seq in zip(pairs, aligned_parents)))
    SeqIO.write(result, outfile, 'fasta')
