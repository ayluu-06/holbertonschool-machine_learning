#!/usr/bin/env python3
"""
Modulo documentado
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    funcion documentada
    """
    reg_losses = tf.stack(model.losses)
    return cost + reg_losses
