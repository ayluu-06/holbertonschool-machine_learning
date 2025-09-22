#!/usr/bin/env python3
"""
Modulo documentado
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    funcion documentada
    """
    l2_sum = 0
    for i in range(1, L + 1):
        l2_sum += np.sum(np.square(weights['W' + str(i)]))
    l2_cost = cost + (lambtha / (2 * m)) * l2_sum
    return l2_cost
