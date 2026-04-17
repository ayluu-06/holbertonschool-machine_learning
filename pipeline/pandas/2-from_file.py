#!/usr/bin/env python3
"""
modulo documentado
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    funcion documentada
    """
    return pd.read_csv(filename, sep=delimiter)
