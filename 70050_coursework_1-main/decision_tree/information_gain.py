# -*- coding: utf-8 -*-

import numpy as np

def entropy(dataset):
    """
    Compute the entropy of the given dataset.

    Parameters:
    - dataset (np.ndarray): The dataset for which entropy is to be calculated.

    Returns:
    - float: Entropy of the dataset.
    """
    
    room_numbers = dataset[:, -1]
    
    total_samples = dataset.shape[0]
    
    # Give an array with counts for samples per room
    _, samples_per_room = np.unique(room_numbers, return_counts=True)
    
    probabilities = samples_per_room / total_samples
    
    # Compute the entropy
    ent = -np.sum(probabilities * np.log2(probabilities))
    
    return ent


def remainder(s_left, s_right):
    """
    Compute the weighted sum (remainder) of the entropies
    of the two subsets, s_left and s_right.

    Parameters:
    - s_left (np.ndarray): The left subset of the dataset, containing
                            sample rows and their labels in the last column.
    - s_right (np.ndarray): The right subset of the dataset, containing
                            sample rows and their labels in the last column.

    Returns:
    - float: Weighted sum of the entropies of the subsets based on their sizes.
    """
    
    total_samples = len(s_left) + len(s_right)
    
    left_entropy_term = (len(s_left) / total_samples) * entropy(s_left)
    right_entropy_term = (len(s_right) / total_samples) * entropy(s_right)
    
    rem = left_entropy_term + right_entropy_term
    
    return rem



def information_gain(s_all, s_left, s_right):
    """
    Compute the information gain.

    Parameters:
    - s_all (np.ndarray): The original dataset before splitting.
    - s_left (np.ndarray): The left subset of the dataset after splitting.
    - s_right (np.ndarray): The right subset of the dataset after splitting.

    Returns:
    - gain (float): Information gain after splitting the dataset.
    """
    gain = entropy(s_all) - remainder(s_left, s_right)
    
    return gain

