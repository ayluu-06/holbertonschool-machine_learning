#!/usr/bin/env python3
"""
0-uni_bleu
"""

import math


def uni_bleu(references, sentence):
    """
    funcion documentada
    """
    sentence_len = len(sentence)

    ref_lens = [len(ref) for ref in references]
    r = min(ref_lens, key=lambda x: abs(x - sentence_len))

    if sentence_len > r:
        bp = 1
    else:
        bp = math.exp(1 - r / sentence_len)

    counts = {}
    for word in sentence:
        counts[word] = counts.get(word, 0) + 1

    max_counts = {}
    for ref in references:
        ref_count = {}
        for word in ref:
            ref_count[word] = ref_count.get(word, 0) + 1
        for word in ref_count:
            max_counts[word] = max(max_counts.get(word, 0),
                                   ref_count[word])

    clipped_count = 0
    for word in counts:
        clipped_count += min(counts[word],
                             max_counts.get(word, 0))

    precision = clipped_count / sentence_len

    return bp * precision
