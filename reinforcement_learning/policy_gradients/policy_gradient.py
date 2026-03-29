#!/usr/bin/env python3
"""
modulo documentado
"""
import numpy as np


def policy(matrix, weight):
    """
    funcion documentada
    """
    z = np.matmul(matrix, weight)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    funcion documentada
    """
    state = state[np.newaxis, :]
    probs = policy(state, weight)
    action = np.random.choice(len(probs[0]), p=probs[0])

    dsoftmax = -probs.copy()
    dsoftmax[0, action] += 1

    grad = np.matmul(state.T, dsoftmax)

    return action, grad
