#!/usr/bin/env python3
"""
modulo documentado
"""

index = __import__('10-index').index


def concat(df1, df2):
    """
    funcion documentada
    """
    df1 = index(df1)
    df2 = index(df2)

    df2 = df2[df2.index <= 1417411920]

    return pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
