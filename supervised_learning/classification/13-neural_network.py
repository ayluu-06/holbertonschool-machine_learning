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

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        funcion documentada
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """
        funcion documentada
        """
        _, A2 = self.forward_prop(X)
        prediction = (A2 >= 0.5).astype(int)
        cost = self.cost(Y, A2)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        funcion documentada
        """
        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = (dZ2 @ A1.T) / m
        db2 = np.sum(dZ2) / m

        dZ1 = (self.__W2.T @ dZ2) * (A1 * (1 - A1))
        dW1 = (dZ1 @ X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1
