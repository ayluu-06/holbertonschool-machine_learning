#!/usr/bin/env python3
"""
0-bag_of_words module
"""

import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    funcion documentada
    """
    tokenized = []
    for sentence in sentences:
        words = re.findall(r"[a-z']+", sentence.lower())
        tokenized.append(words)

    if vocab is None:
        vocab = sorted(set(word for sent in tokenized for word in sent))
    else:
        vocab = list(vocab)

    features = np.array(vocab)

    word_to_index = {word: i for i, word in enumerate(features)}

    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, sent in enumerate(tokenized):
        for word in sent:
            if word in word_to_index:
                embeddings[i, word_to_index[word]] += 1

    return embeddings, features
