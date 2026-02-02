#!/usr/bin/env python3
"""
1-ngram_bleu
"""

import math


def _ngrams(words, n):
    """
    funcion documentada
    """
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def ngram_bleu(references, sentence, n):
    """
    funcion documentada
    """
    c = len(sentence)

    ref_lens = [len(ref) for ref in references]
    r = min(ref_lens, key=lambda x: abs(x - c))

    if c > r:
        bp = 1
    else:
        bp = math.exp(1 - r / c)

    cand_ngrams = _ngrams(sentence, n)
    if len(cand_ngrams) == 0:
        return 0

    cand_counts = {}
    for ng in cand_ngrams:
        cand_counts[ng] = cand_counts.get(ng, 0) + 1

    max_ref_counts = {}
    for ref in references:
        ref_ngrams = _ngrams(ref, n)
        ref_counts = {}
        for ng in ref_ngrams:
            ref_counts[ng] = ref_counts.get(ng, 0) + 1
        for ng in ref_counts:
            max_ref_counts[ng] = max(max_ref_counts.get(ng, 0),
                                     ref_counts[ng])

    clipped = 0
    for ng in cand_counts:
        clipped += min(cand_counts[ng], max_ref_counts.get(ng, 0))

    precision = clipped / len(cand_ngrams)

    return bp * precision
