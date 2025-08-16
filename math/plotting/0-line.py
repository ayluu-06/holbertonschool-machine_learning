#!/usr/bin/env python3
"""
0-line.py
Module that defines a function to plot y = x^3 as a red line
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    y is plotted as a solid red line, x-axis ranges from 0 to 10
    """

    y = np.arange(0, 11) ** 3  # y = x^3 Podria ser y = x ** 3

    plt.figure(figsize=(6.4, 4.8))  # Tamaño de la figura
    plt.xlim(0, 10)  # x va estrictamente de 0 a 10
    plt.plot(y, 'r-')  # 'r-' define la linea roja solida
    plt.show()  # Muestra la grafica
