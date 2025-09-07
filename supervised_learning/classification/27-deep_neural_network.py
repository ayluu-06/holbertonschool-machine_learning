#!/usr/bin/env python3
"""
Module documented
"""
import numpy as np
import pickle
import os


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

            if i < self.__L:
                A = 1.0 / (1.0 + np.exp(-Z))
            else:
                Z_shift = Z - np.max(Z, axis=0, keepdims=True)
                expZ = np.exp(Z_shift)
                A = expZ / np.sum(expZ, axis=0, keepdims=True)

            self.__cache[f"A{i}"] = A
            A_prev = A

        return A_prev, self.__cache

    def cost(self, Y, A):
        """
        funcion documentada
        """
        m = Y.shape[1]
        eps = 1e-8
        cost = -(1 / m) * np.sum(Y * np.log(A + eps))
        return cost

    def evaluate(self, X, Y):
        """
        funcion documentada
        """
        AL, _ = self.forward_prop(X)
        m = AL.shape[1]
        Y_pred = np.zeros_like(AL)
        Y_pred[np.argmax(AL, axis=0), np.arange(m)] = 1
        cost = self.cost(Y, AL)
        return Y_pred, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        funcion documentada
        """
        m = Y.shape[1]
        L = self.__L

        dZ = cache[f"A{L}"] - Y

        for layer in range(L, 0, -1):
            A_prev = cache[f"A{layer-1}"]
            Wl = self.__weights[f"W{layer}"]

            dW = (dZ @ A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            self.__weights[f"W{layer}"] = Wl - alpha * dW
            self.__weights[f"b{layer}"] = (
                self.__weights[f"b{layer}"] - alpha * db)

            if layer > 1:
                A_prev_act = A_prev
                dZ = (Wl.T @ dZ) * (A_prev_act * (1 - A_prev_act))

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=False, graph=False, step=100):
        """
        funcion documentada
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations + 1):
            AL, cache = self.forward_prop(X)

            if verbose and (i % step == 0 or i == iterations):
                cost = self.cost(Y, AL)
                print(f"Cost after {i} iterations: {cost}")

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        funcion documentada
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        funcion documentada
        """
        if not os.path.exists(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)
