# -*- coding: utf-8 -*-

import numpy as np

from utilities.dataset_loader import load_dataset
from training_and_evaluation.k_fold_splitting import split_dataset_into_k_folds
from training_and_evaluation.k_fold_training import k_fold_training
from training_and_evaluation.evaluation_utilities import (
    compute_conf_matrix, compute_norm_conf_matrix, predict_array,
    print_metrics_for_conf_matrix)
from decision_tree.leaf_counter import tree_depth

def task_3(dataset):
    """
    Trains and evaluates decision trees using k-fold cross-validation and 
    prints the metrics for the combined results.
    
    Inputs:
    - dataset (np.ndarray): Data with features and labels.
    
    Outputs:
    - No return value. The metrics are printed directly to the console.
    """
    
    # Find 10 folds of indices for k-fold cross-validation
    folds = split_dataset_into_k_folds(10, len(dataset))
    
    # Train decision trees on each fold and return the trees and test data
    tree_list, test_data_list = k_fold_training(dataset, folds)
    
    # Compute the depth of each trained decision tree
    list_of_tree_depth = [tree_depth(tree) for tree in tree_list]
    
    # Calculate the average depth of the decision trees
    av_tree_depth = np.mean(list_of_tree_depth)
    
    list_of_normalized_confusion_matrices = []
    
    # Predict and compute the confusion matrix for each test data and 
    # its decision tree
    for f_index, (tree, test_data) in enumerate(
            zip(tree_list, test_data_list)):
        actual_labels = test_data[:, -1]
        predicted_labels = predict_array(test_data, tree)
        conf_matrix = compute_conf_matrix(actual_labels, predicted_labels)
        normalized_conf_matrix = compute_norm_conf_matrix(conf_matrix)
        list_of_normalized_confusion_matrices.append(normalized_conf_matrix)
        
    # Average out the normalized confusion matrices from all folds
    av_confusion_matrix = np.mean(list_of_normalized_confusion_matrices, 
                                  axis=0)
    
    # Print computed metrics for the average confusion matrix 
    # and the average tree depth
    print_metrics_for_conf_matrix(av_confusion_matrix, av_tree_depth)
    
    return

if __name__ == "__main__":
    # Load the clean dataset for processing
    dataset_clean = load_dataset("data/clean_dataset.txt")
    
    # Run the main task with the loaded dataset
    task_3(dataset_clean)
