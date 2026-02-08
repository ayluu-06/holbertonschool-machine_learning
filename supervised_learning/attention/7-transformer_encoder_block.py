#!/usr/bin/env python3
"""
modulo documentado
"""

import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    funcion documentada
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        funcion documentada
        """
        super().__init__()

        self.mha = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        funcion documentada
        """
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        hidden_out = self.dense_hidden(out1)
        ffn_out = self.dense_output(hidden_out)
        ffn_out = self.dropout2(ffn_out, training=training)
        out2 = self.layernorm2(out1 + ffn_out)

        return out2
