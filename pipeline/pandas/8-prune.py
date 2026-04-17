#!/usr/bin/env python3
"""
modulo documentado
"""


def prune(df):
    """
    funcion documentada
    """
    return df[df["Close"].notna()]
