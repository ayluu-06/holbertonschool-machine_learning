#!/usr/bin/env python3
"""
Module documented
"""


def minor(matrix):
    """
    funcion documentada
    """
    if not isinstance(matrix, list) or (
        matrix != [] and not all(isinstance(r, list) for r in matrix)
    ):
        raise TypeError("matrix must be a list of lists")

    if matrix == [] or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    if n == 1:
        return [[1]]

    def determinant(m):
        """
        funcion documentada
        """
        if m == [[]]:
            return 1
        if any(len(r) != len(m) for r in m):
            raise ValueError("matrix must be a square matrix")
        size = len(m)
        if size == 1:
            return m[0][0]
        if size == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]
        det = 0
        for j in range(size):
            sub = [row[:j] + row[j+1:] for row in m[1:]]
            det += ((-1) ** j) * m[0][j] * determinant(sub)
        return det

    minors = []
    for i in range(n):
        row = []
        for j in range(n):
            sub = [
                matrix[x][:j] + matrix[x][j+1:]
                for x in range(n) if x != i
            ]
            row.append(determinant(sub))
        minors.append(row)

    return minors
