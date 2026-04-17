#!/usr/bin/env python3
"""
modulo documentado
"""
import pandas as pd


def array(df):
    """
    funcion documentada
    """
    return df[["High", "Close"]].tail(10).to_numpy()
