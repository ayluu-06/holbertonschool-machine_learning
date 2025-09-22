#!/usr/bin/env python3
"""
Modulo documentado
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    funcion documentada
    """
    if opt_cost - cost > threshold:
        return False, 0
    count += 1
    if count >= patience:
        return True, count
    return False, count
