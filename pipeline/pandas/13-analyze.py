#!/usr/bin/env python3
"""
modulo documentado
"""


def analyze(df):
    """
    funcion documentada
    """
    return df.drop(columns=["Timestamp"]).describe()
