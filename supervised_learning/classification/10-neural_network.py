#!/usr/bin/env python3
"""
Module documented
"""
import numpy as np


class NeuralNetwork:
    """
    clase documentada
    """
    def __init__(self, nx, nodes):
        """
        funcion documentada
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        funcion documentada
        """
        return self.__W1

    @property
    def b1(self):
        """
        funcion documentada
        """
        return self.__b1

    @property
    def A1(self):
        """
        funcion documentada
        """
        return self.__A1

    @property
    def W2(self):
        """
        funcion documentada
        """
        return self.__W2

    @property
    def b2(self):
        """
        funcion documentada
        """
        return self.__b2

    @property
    def A2(self):
        """
        funcion documentada
        """
        return self.__A2

    def forward_prop(self, X):
        """
        funcion documentada
        """
        Z1 = self.__W1 @ X + self.__b1
        A1 = 1.0 / (1.0 + np.exp(-Z1))
        self.__A1 = A1

        Z2 = self.__W2 @ A1 + self.__b2
        A2 = 1.0 / (1.0 + np.exp(-Z2))
        self.__A2 = A2

        return self.__A1 self.__A2
