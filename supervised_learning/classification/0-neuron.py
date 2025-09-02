#!/usr/bin/env python3
"""
Module documented
"""
import numpy as np


class Neuron:
    """
    clase documentada
    """
    def __init__(self, nx):
        """
        funcion documentada
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
