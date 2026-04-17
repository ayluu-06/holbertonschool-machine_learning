#!/usr/bin/env python3
"""
modulo documentado
"""
import pandas as pd


def analyze(df):
    """
    funcion documentada
    """
    return df.drop(columns=["Timestamp"]).describe()
