#!/usr/bin/env python3
"""
Modulo documentado
"""
import numpy as np


def precision(confusion):
    """
    funcion documentada
    """
    true_positives = np.diag(confusion)
    predicted = np.sum(confusion, axis=0)
    return true_positives / predicted
