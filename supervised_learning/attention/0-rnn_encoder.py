#!/usr/bin/env python3
"""
0-rnn_encoder module
"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    clase documentada
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        funcion docuemntada
        """
        super().__init__()
        self.batch = batch
        self.units = units

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )

        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """
        funcion documentada
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        funcion documentada
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
