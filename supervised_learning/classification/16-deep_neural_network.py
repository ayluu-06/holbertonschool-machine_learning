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
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(1, self.L + 1):
            layer_size = layers[i - 1]
            prev_size = nx if i == 1 else layers[i - 2]
            self.weights[f"W{i}"] = (np.random.randn(layer_size, prev_size)
                                     * np.sqrt(2 / prev_size))
            self.weights[f"b{i}"] = np.zeros((layer_size, 1))