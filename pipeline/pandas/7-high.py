#!/usr/bin/env python3
"""
modulo documentado
"""
import pandas as pd


def high(df):
    """
    funcion documentada
    """
    return df.sort_values(by="High", ascending=False)
