#!/usr/bin/env python3
"""
Modulo documentado
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    funcion documentada
    """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
