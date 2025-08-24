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

    def update_bounds_below(self):
        """
        función documentada
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        for child in [self.left_child, self.right_child]:  # abajo
            if child is None:
                continue

            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

            f = self.feature
            t = self.threshold
            if child is self.left_child:
                child.lower[f] = max(child.lower.get(f, -np.inf), t)
            else:
                child.upper[f] = min(child.upper.get(f,  np.inf), t)

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()

    def update_indicator(self):
        """
        This method computes the indicator function from the Node.lower and
        Nodeupper dictionaries and stores it in an attribute Nodeindicator
        """
        def is_large_enough(x):
            """
            This function returns a 1D numpy array of size
            `n_individuals` so that the `i`-th element of the later is `True`
            if the `i`-th individual has all its features > the lower bounds.
            """
            return np.array([np.greater(x[:, key], self.lower[key])
                             for key in list(self.lower.keys())]).all(axis=0)

        def is_small_enough(x):
            """
            This function returns a 1D numpy array of size
            `n_individuals` so that the `i`-th element of the later is `True`
            if the `i`-th individual has all its features <= the lower bounds.
            """
            return np.array([np.less_equal(x[:, key], self.upper[key])
                             for key in list(self.upper.keys())]).all(axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
            )

    def pred(self, x):
        """
        funcion documentada
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


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

    def update_bounds_below(self):
        """
        función documentada
        """
        pass

    def pred(self, x):
        """
        funcion documentada
        """
        return self.value


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

    def update_bounds(self):
        """
        función documentada
        """
        self.root.update_bounds_below()

    def update_predict(self):
        """
        funcion documentada
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum([leaf.indicator(A) * leaf.value
                                        for leaf in leaves], axis=0)

    def pred(self, x):
        """
        funcion documentada
        """
        return self.root.pred(x)

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

    def fit(self, explanatory, target, verbose=0):
        """
        funcion documentada
        """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        elif self.split_criterion == "Gini":
            self.split_criterion = self.Gini_split_criterion
        else:
            def _notimpl(node):
                """
                funcion documentada
                """
                raise NotImplementedError("Criterio no implementado")
            self.split_criterion = _notimpl

        self.explanatory = explanatory
        self.target = target

        self.root.sub_population = np.ones_like(self.target, dtype=bool)

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : { self.depth() }
    - Number of nodes           : { self.count_nodes() }
    - Number of leaves          : { self.count_nodes(only_leaves=True) }
    - Accuracy on training data : { self.accuracy(
        self.explanatory,
        self.target
        ) }""")

    def fit_node(self, node):
        """
        funcion documentada
        """
        node.feature, node.threshold = self.split_criterion(node)

        go_left = self.explanatory[:, node.feature] > node.threshold
        left_population = np.logical_and(go_left, node.sub_population)
        right_population = np.logical_and(~go_left, node.sub_population)

        left_count = np.sum(left_population)
        right_count = np.sum(right_population)
        next_depth = node.depth + 1

        def is_pure(pop):
            """
            funcion documentada
            """
            vals = self.target[pop]
            if vals.size == 0:
                return True
            return np.unique(vals).size == 1

        is_left_leaf = (
            left_count < self.min_pop
            or next_depth >= self.max_depth
            or is_pure(left_population)
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (
            right_count < self.min_pop
            or next_depth >= self.max_depth
            or is_pure(right_population)
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """
        funcion documentada
        """
        vals, counts = np.unique(
            self.target[sub_population], return_counts=True)
        value = int(vals[np.argmax(counts)]) if vals.size > 0 else 0
        leaf_child = Leaf(value)
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

    def accuracy(self, test_explanatory, test_target):
        """
        funcion documentada
        """
        return np.sum(
            self.predict(test_explanatory) == test_target) / test_target.size

    def possible_thresholds(self, node, feature):
        """
        funcion documentada
        """
        vals = np.unique(self.explanatory[:, feature][node.sub_population])
        if vals.size <= 1:
            return np.array([], dtype=float)
        return (vals[1:] + vals[:-1]) / 2.0

    def Gini_split_criterion_one_feature(self, node, feature):
        """
        funcion documentada
        """
        mask = node.sub_population
        Xf = self.explanatory[:, feature][mask]
        y = self.target[mask]
        th = self.possible_thresholds(node, feature)
        t = th.size
        if t == 0:
            return np.array([0.0, 1.0], dtype=float)
        classes = np.unique(y)
        class_match = (y[:, None] == classes[None, :])
        left_cmp = (Xf[:, None] > th[None, :])
        Left_F = class_match[:, None, :] & left_cmp[:, :, None]
        L_counts = Left_F.sum(axis=0)
        n_left = L_counts.sum(axis=1)
        C_counts = class_match.sum(axis=0)
        R_counts = C_counts[None, :] - L_counts
        n_right = C_counts.sum() - n_left
        with np.errstate(divide='ignore', invalid='ignore'):
            p_left = np.where(
                n_left[:, None] > 0, L_counts / n_left[:, None], 0.0)
        gini_left = 1.0 - np.sum(p_left * p_left, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            p_right = np.where(
                n_right[:, None] > 0, R_counts / n_right[:, None], 0.0)
        gini_right = 1.0 - np.sum(p_right * p_right, axis=1)
        gini_avg = (
            n_left * gini_left + n_right * gini_right
            ) / (n_left + n_right)
        j = np.argmin(gini_avg)
        return np.array([th[j], gini_avg[j]], dtype=float)

    def Gini_split_criterion(self, node):
        """
        funcion documentada
        """
        X = np.array([
            self.Gini_split_criterion_one_feature(node, i)
            for i in range(self.explanatory.shape[1])
        ])
        i = np.argmin(X[:, 1])
        return int(i), float(X[i, 0])


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
