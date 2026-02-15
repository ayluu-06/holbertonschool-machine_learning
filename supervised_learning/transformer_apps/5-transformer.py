#!/usr/bin/env python3
"""
clase documentado
"""

import tensorflow as tf


def _positional_encoding(max_seq_len, dm):
    """funcion documentada"""
    pos = tf.cast(tf.range(max_seq_len)[:, tf.newaxis], tf.float32)
    i = tf.cast(tf.range(dm)[tf.newaxis, :], tf.float32)

    angle_rates = 1.0 / tf.pow(
        10000.0, (2.0 * (i // 2.0)) / tf.cast(dm, tf.float32))
    angles = pos * angle_rates

    sin = tf.sin(angles[:, 0::2])
    cos = tf.cos(angles[:, 1::2])

    pe = tf.concat([sin, cos], axis=-1)
    pe = pe[:, :dm]
    return pe[tf.newaxis, :, :]


def _scaled_dot_product_attention(q, k, v, mask):
    """funcion documentada"""
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_logits += (mask * -1e9)

    weights = tf.nn.softmax(scaled_logits, axis=-1)
    output = tf.matmul(weights, v)
    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """clase documentada"""

    def __init__(self, dm, h):
        """funcion documentada"""
        super().__init__()
        self.dm = dm
        self.h = h
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def _split_heads(self, x, batch_size):
        """Split last dim into (h, depth) and transpose to attention format."""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=(0, 2, 1, 3))

    def call(self, Q, K, V, mask=None):
        """funcion documentada"""
        batch_size = tf.shape(Q)[0]

        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        q = self._split_heads(q, batch_size)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        attn_output, weights = _scaled_dot_product_attention(q, k, v, mask)

        attn_output = tf.transpose(attn_output, perm=(0, 2, 1, 3))
        concat = tf.reshape(attn_output, (batch_size, -1, self.dm))

        output = self.linear(concat)
        return output, weights


class EncoderBlock(tf.keras.layers.Layer):
    """clase documentada"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """funcion documentada"""
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """funcion documentada"""
        attn_out, _ = self.mha(x, x, x, mask)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.layernorm1(x + attn_out)

        ffn = self.dense_hidden(out1)
        ffn = self.dense_output(ffn)
        ffn = self.dropout2(ffn, training=training)
        out2 = self.layernorm2(out1 + ffn)

        return out2


class DecoderBlock(tf.keras.layers.Layer):
    """clase documentada"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """funcion documentada"""
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output,
             training, look_ahead_mask=None, padding_mask=None):
        """funcion documentada"""
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2, _ = self.mha2(
            out1, encoder_output, encoder_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        ffn = self.dense_hidden(out2)
        ffn = self.dense_output(ffn)
        ffn = self.dropout3(ffn, training=training)
        out3 = self.layernorm3(out2 + ffn)

        return out3


class Encoder(tf.keras.layers.Layer):
    """clase documentada"""

    def __init__(self, N, dm, h, hidden,
                 input_vocab, max_seq_len, drop_rate=0.1):
        """funcion documentada"""
        super().__init__()
        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = _positional_encoding(max_seq_len, dm)

        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
            ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """funcion documentada"""
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        x += self.positional_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    """clase documentada"""

    def __init__(self, N, dm, h, hidden,
                 target_vocab, max_seq_len, drop_rate=0.1):
        """Initialize decoder."""
        super().__init__()
        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = _positional_encoding(max_seq_len, dm)

        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
            ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output,
             training, look_ahead_mask=None, padding_mask=None):
        """funcion documentada"""
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        x += self.positional_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, encoder_output,
                      training, look_ahead_mask, padding_mask)

        return x


class Transformer(tf.keras.Model):
    """funcion documentada"""

    def __init__(
        self,
        N,
        dm,
        h,
        hidden,
        input_vocab,
        target_vocab,
        max_seq_input,
        max_seq_target,
        drop_rate=0.1,
    ):
        """funcion documentada"""
        super().__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate
        )
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate
        )
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(
        self,
        inputs,
        target,
        training,
        encoder_mask=None,
        look_ahead_mask=None,
        decoder_mask=None,
    ):
        """Forward pass."""
        enc_out = self.encoder(inputs, training, encoder_mask)
        dec_out = self.decoder(
            target, enc_out, training, look_ahead_mask, decoder_mask
        )
        return self.linear(dec_out)
