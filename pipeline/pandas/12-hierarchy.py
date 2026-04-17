#!/usr/bin/env python3
"""
modulo documentado
"""

index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    funcion documentada
    """
    df1 = index(df1)
    df2 = index(df2)

    df1 = df1[(df1.index >= 1417411980) & (df1.index <= 1417417980)]
    df2 = df2[(df2.index >= 1417411980) & (df2.index <= 1417417980)]

    df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
    df = df.swaplevel(0, 1)
    df = df.sort_index()

    return df
