#!/usr/bin/env python3
"""
modulo documentado
"""
import pandas as pd


def prune(df):
    """
    funcion documentada
    """
    return df[df["Close"].notna()]
