#!/usr/bin/env python3
"""
Module to calculate the sum of squares from 1 to n
"""


def summation_i_squared(n):
    """
    Return the sum of squares from 1 to n
    """
    if not isinstance(n, int) or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
