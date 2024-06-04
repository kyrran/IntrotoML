# -*- coding: utf-8 -*-
import numpy as np

from training_and_evaluation.k_fold_splitting import k_fold_split
from decision_tree.decision_tree import decision_tree_learning
from training_and_evaluation.evaluation_utilities import (
    predict_array,
    compute_conf_matrix,
    compute_norm_conf_matrix
)
from pruning.pruning import pruning_single_tree
from decision_tree.leaf_counter import tree_depth

def nested_cross_validation_pruning(dataset, n_folds):
    """
    Perform nested cross-validation on the dataset and apply pruning to 
    decision trees. This function uses two layers of cross-validation: an 
    outer loop for testing and an inner loop for validation. The decision 
    trees are trained on training data, pruned using validation data, and 
    tested on test data.

    Parameters:
    - dataset (np.ndarray): Dataset for nested cross-validation.
    - n_folds (int): Number of folds for outer and inner cross-validation.

    Returns:
    - list of np.ndarray: Normalized confusion matrices for unpruned trees.
    - list of np.ndarray: Normalized confusion matrices for pruned trees.
    - float: Average depth of unpruned trees across all iterations.
    - float: Average depth of the pruned trees across all iterations.
    """
    
    # Split the dataset indices into k-folds
    folds_split_indices = k_fold_split(n_folds, len(dataset))
    
    # Lists to store results for pruned and unpruned trees
    pruned_results = []
    og_results = []
    
    # Lists to store the depth of pruned and unpruned trees
    pruned_tree_depth_array = []
    og_tree_depth_array = []
    
    # Initial list for the outer cross-validation loop
    folds_outer_loop = list(range(n_folds))
    
    # Begin nested cross-validation
    for fold_index_outer in folds_outer_loop:
        
        # Create list for inner CV loop, excluding current fold of outer loop
        folds_inner_loop = list(range(n_folds))
        folds_inner_loop.pop(fold_index_outer)        
        
        # Extract test data for current fold of outer loop
        test_data_indices = folds_split_indices[fold_index_outer]
        test_data = dataset[test_data_indices,:]

        # Inner loop for cross-validation
        for fold_index_inner in folds_inner_loop:
            
            # Extract validation data for current fold of inner loop
            val_data_indices = folds_split_indices[fold_index_inner]
            val_data = dataset[val_data_indices,:]
            
            # Determine indices for training data
            folds_training_index = list(range(n_folds))
            excluded_indices = [fold_index_outer, fold_index_inner]
            folds_training_index = [
                item for idx, item in enumerate(folds_training_index)
                if idx not in excluded_indices
            ]
            
            # Extract training data using determined indices
            training_data_indices = np.concatenate(
                [folds_split_indices[i] for i in folds_training_index]
            )
            training_data = dataset[training_data_indices,:]

            # Train an unpruned decision tree
            decision_tree = decision_tree_learning(training_data)
            
            # Apply pruning using the validation dataset
            og_tree = pruning_single_tree(decision_tree, val_data)
            pruned_tree = decision_tree
            
            # Predict labels for test data using both trees
            actual_labels = test_data[:, -1]
            og_predicted_labels = predict_array(test_data, og_tree)
            pruned_predicted_labels = predict_array(test_data, pruned_tree)
            
            # Compute confusion matrices for both sets of predictions
            og_conf_matrix = compute_conf_matrix(actual_labels, 
                                                 og_predicted_labels)
            og_norm_conf_matrix = compute_norm_conf_matrix(og_conf_matrix)
            
            pruned_conf_matrix = compute_conf_matrix(actual_labels, 
                                                     pruned_predicted_labels)
            pruned_norm_conf_matrix = compute_norm_conf_matrix(
                pruned_conf_matrix)
            
            # Store results for later analysis
            og_results.append(og_norm_conf_matrix)
            pruned_results.append(pruned_norm_conf_matrix)

            # Calculate and store the depth of trees
            pruned_tree_depth = tree_depth(pruned_tree)
            pruned_tree_depth_array.append(pruned_tree_depth)
            og_tree_depth = tree_depth(og_tree)
            og_tree_depth_array.append(og_tree_depth)
            
    # Calculate average depth across all trees
    av_og_tree_depth = np.mean(og_tree_depth_array)
    av_pruned_tree_depth = np.mean(pruned_tree_depth_array)   
    
    return og_results, pruned_results, av_og_tree_depth, av_pruned_tree_depth
