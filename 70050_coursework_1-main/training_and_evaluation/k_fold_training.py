# -*- coding: utf-8 -*-
import numpy as np

from decision_tree.decision_tree import decision_tree_learning

def k_fold_training(dataset, folds):
    """
    Train k models on the k training sets and return all models

    Train k models on the k training sets (training and validation sets)

    Parameters:
        - dataset (np.ndarray): Dataset to analyze
        - folds (list of k elements): Each element is a list of indices

    Returns:
        - tree_list (list): list of dictionaries with decision trees in the 
                            order of folds
        - test_data_list (list): list of test_data which are in the format
                                    of np.array
    """

    tree_list = []
    test_data_list = []

    # Training for the models
    for fold_index in range(len(folds)):
        
        training_data_indices = np.concatenate((folds[fold_index][0],
                                                folds[fold_index][1]), 
                                               axis=None)
        training_data = dataset[training_data_indices,:]
        
        # Appending test_data
        test_data_indices = folds[fold_index][2]
        test_data = dataset[test_data_indices,:]
        test_data_list.append(test_data)

        tree = decision_tree_learning(training_data)
        tree_list.append(tree)

    return tree_list, test_data_list