#!/usr/bin/env python3
"""
modulo documentado
"""
import pandas as pd


def slice(df):
    """
    funcion documentada
    """
    return df[["High", "Low", "Close", "Volume_(BTC)"]][::60]
