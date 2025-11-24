#!/usr/bin/env python3
"""
Module documented
"""


def determinant(matrix):
    """
    funcion documentada
    """
    if not isinstance(matrix, list) or (
        matrix != [] and not all(isinstance(r, list) for r in matrix)
    ):
        raise TypeError("matrix must be a list of lists")

    if matrix == []:
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(n):
        sub = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(sub)

    return det
