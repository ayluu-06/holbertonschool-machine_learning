#!/usr/bin/env python3
"""
Modulo documentado
"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    funcion documentada
    """
    return tf.keras.optimizers.SGD(
        learning_rate=alpha, momentum=beta1, nesterov=False)
