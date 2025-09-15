#!/usr/bin/env python3
"""
Modulo documentado
"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    funcion documentada
    """
    return tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
