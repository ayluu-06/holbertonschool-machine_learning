#!/usr/bin/env python3
"""
3-gensim_to_keras
"""

import tensorflow as tf


def gensim_to_keras(model):
    """
    funcion documentada
    """
    weights = model.wv.vectors
    vocab_size, vector_size = weights.shape

    layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[weights],
        trainable=True
    )
    return layer
