#!/usr/bin/env python3
"""
Modulo documentado
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    funcion documentada
    """
    v = beta1 * v + grad
    var = var - alpha * v
    return var, v
