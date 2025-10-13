#!/usr/bin/env python3
"""
Modulo documentado
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    funcion documentada
    """
    m, classes = labels.shape
    y_true_idx = np.argmax(labels, axis=1)
    y_pred_idx = np.argmax(logits, axis=1)
    
    confusion = np.zeros((classes, classes), dtype=int)
    
    for k in range(m):
        i = y_true_idx[k]
        j = y_pred_idx[k]
        confusion[i, j] += 1

    return confusion
