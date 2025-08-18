#!/usr/bin/env python3
"""
Module to calculate the derivative of a polynomial
"""


def poly_derivative(poly):
    """
    Calculate the derivative of a polynomial represented by coefficients list
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None

    if len(poly) == 1:
        return [0]

    derivative = [i * poly[i] for i in range(1, len(poly))]

    return derivative
