# -*- coding: utf-8 -*-

import numpy as np

from numpy.random import default_rng

def split_dataset_into_k_folds(n_folds, n_instances, 
                               random_generator=default_rng()):
    """ Generate train and test indices at each fold.

    Args:
    - n_folds (int): Number of folds
    - n_instances (int): Total number of instances
    - random_generator (np.random.Generator): A random generator

    Returns:
    - list: a list of length n_folds. Each element in the list is a list
            (or tuple) with three elements:
                - a numpy array containing the train indices
                - a numpy array containing the val indices
                - a numpy array containing the test indices
    """

    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        # pick k as test, and k+1 as validation (or 0 if k is the final split)
        test_indices = split_indices[k]
        val_indices = split_indices[(k + 1) % n_folds]

        # concatenate remaining splits for training
        train_indices = np.zeros((0,), dtype=int)
        for i in range(n_folds):
            # concatenate to training set if not validation or test
            if i not in [k, (k + 1) % n_folds]:
                train_indices = np.hstack([train_indices, split_indices[i]])

        folds.append([train_indices, val_indices, test_indices])

    return folds


def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    """ Split n_instances into n mutually exclusive splits at random.

    Args:
    - n_splits (int): Number of splits
    - n_instances (int): Number of instances to split
    - random_generator (np.random.Generator): A random generator

    Returns:
    - list: a list (length n_splits). Each element in the list should contain
            a numpy array giving the indices of the instances in that split.
    """

    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices