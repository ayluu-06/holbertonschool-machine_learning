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

    def forward_prop(self, X):
        """
        funcion documentada
        """
        Z = self.__W @ X + self.__b
        self.__A = 1.0 / (1.0 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        funcion documentada
        """
        m = Y.shape[1]

        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        funcion documentada
        """
        A = self.forward_prop(X)
        prediction = (A >= 0.5).astype(int)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        funcion documentada
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = (dZ @ X.T) / m
        db = np.sum(dZ) / m
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        funcion documentada
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")

        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")

        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")

            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        xs = []
        cs = []

        A0 = self.forward_prop(X)
        cost0 = self.cost(Y, A0)
        if verbose or graph:
            xs.append(0)
            cs.append(cost0)
            if verbose:
                print(f"Cost after 0 iterations: {cost0}")

        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

            is_last = (i == iterations - 1)
            if (verbose or graph) and (((i + 1) % step) == 0 or is_last):
                A_now = self.forward_prop(X)
                c_now = self.cost(Y, A_now)
                xs.append(i + 1)
                cs.append(c_now)
                if verbose:
                    print(f"Cost after {i + 1} iterations: {c_now}")

        if graph:
            import matplotlib.pyplot as plt
            plt.plot(xs, cs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
