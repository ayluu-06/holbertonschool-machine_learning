#!/usr/bin/env python3
"""
Module documented
"""
import numpy as np


class Node:
    """
    clase documentada
    """
    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        """
        funcion documentada
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        funcion documentada
        """

        if self.is_leaf:
            return self.depth  # si ya es una hoja se guarda su prof.

        left_depth = (self.left_child.max_depth_below()
                      if self.left_child is not None  # SiTieneHijoIzq,SuProf
                      else self.depth)  # sino, guarda su propia prof.

        right_depth = (self.right_child.max_depth_below()
                       if self.right_child is not None
                       else self.depth)

        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        función documentada
        """
        left_count = (self.left_child.count_nodes_below(only_leaves)
                      if self.left_child is not None
                      else (0))

        right_count = (self.right_child.count_nodes_below(only_leaves)
                       if self.right_child is not None
                       else (0))

        if only_leaves:
            return left_count + right_count
        else:
            return (1 + left_count + right_count)

    def __str__(self):
        """
        función documentada
        """
        header = (f"root [feature={self.feature}, threshold={self.threshold}]"
                  if self.is_root
                  else f"-> node [feature={self.feature}, "
                  f"threshold={self.threshold}]")

        left_text = (self.left_child.__str__()
                     if self.left_child is not None else "")

        left_block = left_child_add_prefix(left_text)

        right_text = (self.right_child.__str__()
                      if self.right_child is not None else "")

        right_block = right_child_add_prefix(right_text)

        return header + "\n" + left_block + right_block

    def get_leaves_below(self):
        """
        función documentada
        """
        left_list = (self.left_child.get_leaves_below()
                     if self.left_child is not None else [])

        right_list = (self.right_child.get_leaves_below()
                      if self.right_child is not None else [])

        return left_list + right_list


class Leaf(Node):
    """
    clase documentada
    """

    def __init__(self, value, depth=None):
        """
        funcion documentada
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def count_nodes_below(self, only_leaves=False):
        """
        función documentada
        """
        return 1

    def max_depth_below(self):
        """
        funcion documentada
        """
        return self.depth

    def __str__(self):
        """
        función documentada
        """
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """
        funcion documentada
        """
        return [self]


class Decision_Tree():
    """
    clase documentada
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        funcion documentada
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def count_nodes(self, only_leaves=False):
        """
        función documentada
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def depth(self):
        """
        función documentada
        """
        return self.root.max_depth_below()

    def __str__(self):
        """
        función documentada
        """
        return self.root.__str__()

    def get_leaves(self):
        """
        funcion documentada
        """
        return self.root.get_leaves_below()


def left_child_add_prefix(text):
    """
    función documentada
    """
    lines = text.split("\n")
    new_text = "    +--" + lines[0] + "\n"
    for x in lines[1:]:
        if x.strip() != "":
            new_text += "    |  " + x + "\n"
    return new_text


def right_child_add_prefix(text):
    """
    función documentada
    """
    lines = text.split("\n")
    new_text = "    +--" + lines[0] + "\n"
    for x in lines[1:]:
        if x.strip() != "":
            new_text += "       " + x + "\n"
    return new_text
