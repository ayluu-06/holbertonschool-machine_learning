#!/usr/bin/env python3
"""
modulo documentado
"""
import numpy as np


def q_init(env):
    """
    funcion documentada
    """
    states = env.observation_space.n
    actions = env.action_space.n

    Q = np.zeros((states, actions))

    return Q
