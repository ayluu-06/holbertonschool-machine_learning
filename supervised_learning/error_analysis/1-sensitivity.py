#!/usr/bin/env python3
"""
Modulo documentado
"""
import numpy as np


def sensitivity(confusion):
    """
    funcion documentada
    """
    true_positives = np.diag(confusion)
    totals = np.sum(confusion, axis=1)
    return true_positives / totals
