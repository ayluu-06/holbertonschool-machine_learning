#!/usr/bin/env python3
"""
Module to plot stacked bar chart of fruit quantities per person
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Function to plot a stacked bar chart showing the number of fruits
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    plt.title("Number of Fruit per Person")

    x = np.arange(3)
    plt.ylabel("Quantity of Fruit")
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))

    plt.bar(x, fruit[0], width=0.5, label="apples", color='red')
    plt.bar(x, fruit[1], bottom=fruit[0], width=0.5,
            label="bananas", color='yellow')
    plt.bar(x, fruit[2], bottom=fruit[0] + fruit[1], width=0.5,
            label="oranges", color='#ff8000')
    plt.bar(x, fruit[3], bottom=fruit[0] + fruit[1] + fruit[2], width=0.5,
            label="peaches", color='#ffe5b4')

    plt.xticks(x, ["Farrah", "Fred", "Felicia"])
    plt.legend()
    plt.show()
