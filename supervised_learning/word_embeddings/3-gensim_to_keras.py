#!/usr/bin/env python3
"""
3-gensim_to_keras module
"""

import keras


def gensim_to_keras(model):
    """
    funcion documentada
    """
    weights = model.wv.vectors
    vocab_size, vector_size = weights.shape

    embedding = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[weights],
        trainable=True
    )

    return embedding
