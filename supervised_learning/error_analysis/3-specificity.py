#!/usr/bin/env python3
"""
Modulo documentado
"""
import numpy as np


def specificity(confusion):
    """
    funcion documentada
    """
    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    false_negatives = np.sum(confusion, axis=1) - true_positives
    true_negatives = np.sum(confusion) - (
        true_positives + false_positives + false_negatives)
    return true_negatives / (true_negatives + false_positives)
