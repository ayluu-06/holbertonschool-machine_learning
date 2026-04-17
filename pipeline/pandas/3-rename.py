#!/usr/bin/env python3
"""
modulo documentado
"""
import pandas as pd


def rename(df):
    """
    funcion documentada
    """
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit='s')
    return df[["Datetime", "Close"]]
