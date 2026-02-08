#!/usr/bin/env python3
"""
2-rnn_decoder module
"""

import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    clase documentada
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        funcion documentada
        """
        super().__init__()
        self.batch = batch
        self.units = units

        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)

        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        funcion documentada
        """
        context, _ = self.attention(s_prev, hidden_states)

        x = self.embedding(x)

        context = tf.expand_dims(context, axis=1)
        x = tf.concat([context, x], axis=-1)

        outputs, s = self.gru(x, initial_state=s_prev)

        outputs = tf.reshape(outputs, (outputs.shape[0], outputs.shape[2]))
        y = self.F(outputs)

        return y, s
