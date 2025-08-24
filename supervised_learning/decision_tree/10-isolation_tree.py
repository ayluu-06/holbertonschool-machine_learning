#!/usr/bin/env python3
"""
Module documentado
"""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """
    class documentada
    """
    def __init__(self, max_depth=10, seed=0, root=None):
        """
        funcion documentada
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """
        funcion documentada
        """
        return self.root.__str__()

    def depth(self):
        """
        funcion documentada
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        funcion documentada
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """
        funcion documentada
        """
        self.root.update_bounds_below()

    def get_leaves(self):
        """
        funcion documentada
        """
        return self.root.get_leaves_below()

    def update_predict(self):
        """
        funcion documentada
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            [leaf.indicator(A) * leaf.value for leaf in leaves], axis=0
        )

    def np_extrema(self, arr):
        """
        funcion documentada
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        funcion documentada
        """
        diff = 0.0
        while diff == 0.0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            col = self.explanatory[:, feature][node.sub_population]
            fmin, fmax = self.np_extrema(col)
            diff = fmax - fmin
        x = self.rng.uniform()
        threshold = (1 - x) * fmin + x * fmax
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """
        funcion documentada
        """
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        funcion documentada
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """
        funcion documentada
        """
        node.feature, node.threshold = self.random_split_criterion(node)

        go_left = self.explanatory[:, node.feature] > node.threshold
        left_population = np.logical_and(go_left, node.sub_population)
        right_population = np.logical_and(~go_left, node.sub_population)

        next_depth = node.depth + 1

        is_left_leaf = (next_depth >= self.max_depth) or (
            np.sum(left_population) <= 1)
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (next_depth >= self.max_depth) or (
            np.sum(right_population) <= 1)
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """
        funcion documentada
        """
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype=bool)
        self.fit_node(self.root)
        self.update_predict()
        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
