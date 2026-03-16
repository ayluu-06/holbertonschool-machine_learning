#!/usr/bin/env python3
"""
modulo documentado
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    funcion documentada
    """
    p = np.random.uniform(0, 1)

    if p < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state])

    return action
