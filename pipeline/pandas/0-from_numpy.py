#!/usr/bin/env python3
"""
modulo documentado
"""
import pandas as pd


def from_numpy(array):
    """
    funcion documentada
    """
    cols = [chr(65 + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=cols)
