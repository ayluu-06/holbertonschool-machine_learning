#!/usr/bin/env python3
"""
Module documented
"""
import numpy as np


class DeepNeuralNetwork:
    """
    clase documentada
    """
    def __init__(self, nx, layers):
        """
        funcion documentada
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not set(map(type, layers)) <= {int}:
            raise TypeError("layers must be a list of positive integers")
        if min(layers) <= 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(1, self.L + 1):
            layer_size = layers[i - 1]
            prev_size = nx if i == 1 else layers[i - 2]
            self.weights[f"W{i}"] = (np.random.randn(layer_size, prev_size)
                                     * np.sqrt(2 / prev_size))
            self.weights[f"b{i}"] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """
        funcion documentada
        """
        return self.__L

    @property
    def cache(self):
        """
        funcion documentada
        """
        return self.__cache

    @property
    def weights(self):
        """
        funcion documentada
        """
        return self.__weights

    def forward_prop(self, X):
        """
        funcion documentada
        """
        self.__cache["A0"] = X
        A_prev = X

        for i in range(1, self.__L + 1):
            W = self.__weights[f"W{i}"]
            b = self.__weights[f"b{i}"]
            Z = W @ A_prev + b
            A = 1.0 / (1.0 + np.exp(-Z))
            self.__cache[f"A{i}"] = A
            A_prev = A

        return A_prev, self.__cache
