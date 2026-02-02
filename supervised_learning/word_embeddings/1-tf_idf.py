#!/usr/bin/env python3
"""
1-tf_idf module
"""

import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """
    funcion documentada
    """
    tokenized = []
    for sentence in sentences:
        s = sentence.lower()
        s = re.sub(r"'s\b", "", s)
        words = re.findall(r"[a-z]+", s)
        tokenized.append(words)

    if vocab is None:
        vocab = sorted(set(word for sent in tokenized for word in sent))
    else:
        vocab = list(vocab)

    features = np.array(vocab)
    vocab_index = {word: i for i, word in enumerate(features)}

    s = len(sentences)
    f = len(features)

    tf = np.zeros((s, f))
    df = np.zeros(f)

    for i, sent in enumerate(tokenized):
        for word in sent:
            if word in vocab_index:
                tf[i, vocab_index[word]] += 1

    for j in range(f):
        df[j] = np.count_nonzero(tf[:, j])

    idf = np.log((1 + s) / (1 + df)) + 1

    tf_idf_matrix = tf * idf

    norms = np.linalg.norm(tf_idf_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    tf_idf_matrix = tf_idf_matrix / norms

    return tf_idf_matrix, features
