#!/usr/bin/env python3
"""
modulo documentado
"""


def array(df):
    """
    funcion documentada
    """
    return df[["High", "Close"]].tail(10).to_numpy()
