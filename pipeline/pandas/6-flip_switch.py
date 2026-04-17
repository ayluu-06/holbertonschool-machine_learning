#!/usr/bin/env python3
"""
modulo documentado
"""
import pandas as pd


def flip_switch(df):
    """
    funcion documentada
    """
    return df.sort_index(ascending=False).transpose()
