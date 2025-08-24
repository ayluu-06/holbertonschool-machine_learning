#!/usr/bin/env python3
"""
Module documented
"""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """
    clase documentada
    """
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
        funcion documentada
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """
        funcion documentada
        """
        preds = []
        for tree_predict in self.numpy_preds:
            preds.append(tree_predict(explanatory))
        preds = np.array(preds)
        final_preds = []
        for i in range(preds.shape[1]):
            vals, counts = np.unique(preds[:, i], return_counts=True)
            final_preds.append(vals[np.argmax(counts)])
        return np.array(final_preds)

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """
        funcion documentada
        """
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth,
                              min_pop=self.min_pop,
                              seed=self.seed + i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {self.accuracy(
        self.explanatory, self.target)}""")

    def accuracy(self, test_explanatory, test_target):
        """
        funcion documentada
        """
        return np.sum(np.equal(self.predict(
            test_explanatory), test_target)) / test_target.size
