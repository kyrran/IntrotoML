# -*- coding: utf-8 -*-

import numpy as np

from utilities.dataset_loader import load_dataset
from training_and_evaluation.evaluation_utilities import ( 
    compute_norm_conf_matrix)
from training_and_evaluation.evaluation_utilities import ( 
    print_metrics_for_conf_matrix)
from pruning.ncv_pruning import nested_cross_validation_pruning

    
def task_4(dataset):
    """
    Performs nested cross-validation pruning on the given dataset and prints 
    metrics for both original and pruned trees.
    
    Parameters:
    - dataset (np.ndarray): The data containing both features and labels.
    
    Outputs:
    - No return value. Metrics are printed directly to the console.
    """
    
    # Perform nested cross-validation pruning and retrieve results
    (og_results, pruned_results, av_og_tree_depth, 
     av_pruned_tree_depth) = nested_cross_validation_pruning(dataset, 10)
    
    # Normalize the confusion matrices of the original trees' results
    og_norm_conf_matrices = [compute_norm_conf_matrix(matrix) 
                             for matrix in og_results]

    # Normalize the confusion matrices of the pruned trees' results
    pruned_norm_conf_matrices = [compute_norm_conf_matrix(matrix) 
                                 for matrix in pruned_results]
    
    # Calculate the average normalized confusion matrix for both types of trees
    og_av_norm_conf_matrix = np.mean(og_norm_conf_matrices, axis=0)
    pruned_av_norm_conf_matrix = np.mean(pruned_norm_conf_matrices, axis=0)
    
    # Print metrics for both types of trees using their average 
    # normalized confusion matrices and average depths
    print("For the unpruned trees we get the following results:")
    print("")
    print_metrics_for_conf_matrix(og_av_norm_conf_matrix, av_og_tree_depth)
    
    print("For the pruned trees we get the following results:")
    print("")
    print_metrics_for_conf_matrix(pruned_av_norm_conf_matrix,
                                  av_pruned_tree_depth)
    
    return

if __name__ == "__main__":
    # Load the clean dataset for processing
    dataset_clean = load_dataset("data/clean_dataset.txt")
    
    # Run the main task with the loaded dataset
    task_4(dataset_clean)
