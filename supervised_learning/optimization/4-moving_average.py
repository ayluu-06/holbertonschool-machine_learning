#!/usr/bin/env python3
"""
Modulo documentado
"""
import numpy as np


def moving_average(data, beta):
    """
    funcion documentada
    """
    m_avg = []
    v = 0
    for t, x in enumerate(data, start=1):
        v = beta * v + (1 - beta) * x
        v_corr = v / (1 - beta ** t)
        m_avg.append(v_corr)
    return m_avg
