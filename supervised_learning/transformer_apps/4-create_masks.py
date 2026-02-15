#!/usr/bin/env python3

"""
modulo documentado
"""

import tensorflow as tf


def create_masks(inputs, target):
    """funcion documentada"""
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    dec_target_padding = tf.cast(tf.math.equal(target, 0), tf.float32)
    dec_target_padding = dec_target_padding[:, tf.newaxis, tf.newaxis, :]

    seq_len_out = tf.shape(target)[1]
    look_ahead = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0)
    look_ahead = tf.cast(look_ahead, tf.float32)
    look_ahead = look_ahead[tf.newaxis, tf.newaxis, :, :]

    combined_mask = tf.maximum(look_ahead, dec_target_padding)

    decoder_mask = encoder_mask

    return encoder_mask, combined_mask, decoder_mask
