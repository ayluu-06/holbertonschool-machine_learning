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
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        funcion documentada
        """
        return self.__W

    @property
    def b(self):
        """
        funcion documentada
        """
        return self.__b

    @property
    def A(self):
        """
        funcion documentada
        """
        return self.__A
