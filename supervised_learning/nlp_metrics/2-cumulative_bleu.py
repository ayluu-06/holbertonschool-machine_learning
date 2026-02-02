#!/usr/bin/env python3
"""
2-cumulative_bleu
"""

import math


def _ngrams(words, n):
    """
    funcion documentada
    """
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def cumulative_bleu(references, sentence, n):
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

    log_precisions = []

    for k in range(1, n + 1):
        cand_ngrams = _ngrams(sentence, k)
        if len(cand_ngrams) == 0:
            return 0

        cand_counts = {}
        for ng in cand_ngrams:
            cand_counts[ng] = cand_counts.get(ng, 0) + 1

        max_ref_counts = {}
        for ref in references:
            ref_ngrams = _ngrams(ref, k)
            ref_counts = {}
            for ng in ref_ngrams:
                ref_counts[ng] = ref_counts.get(ng, 0) + 1
            for ng in ref_counts:
                max_ref_counts[ng] = max(
                    max_ref_counts.get(ng, 0),
                    ref_counts[ng]
                )

        clipped = 0
        for ng in cand_counts:
            clipped += min(cand_counts[ng],
                           max_ref_counts.get(ng, 0))

        precision = clipped / len(cand_ngrams)
        if precision == 0:
            return 0

        log_precisions.append(math.log(precision))

    avg_log_precision = sum(log_precisions) / n

    return bp * math.exp(avg_log_precision)
