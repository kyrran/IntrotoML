# -*- coding: utf-8 -*-

import numpy as np
from decision_tree.information_gain import information_gain

def split_dataset(dataset, attribute_index, value):
    """
    Splits the dataset based on a given attribute and its value.

    Parameters:
    - dataset (np.ndarray): Dataset to split.
    - attribute_index (int): Index of the attribute for dataset splitting.
    - value (float): Value of the attribute for splitting.

    Returns:
    - tuple: (left_split_dataset, right_split_dataset)
    """
    left_split = dataset[dataset[:, attribute_index] <= value]
    right_split = dataset[dataset[:, attribute_index] > value]
    
    return left_split, right_split

def find_split(dataset):
    """
    Identify the best splitting point based on maximum information gain.

    Parameters:
    - dataset (np.ndarray): Dataset to analyze.

    Returns:
    - tuple: (best_attribute_index, best_value, highest_gain)
    """
    highest_gain = -float('inf')
    best_attribute_index = -1
    best_value = None

    # Exclude the label column
    number_of_attributes = dataset.shape[1] - 1
    
    for attribute_index in range(number_of_attributes):
        # Extract unique values of the column to check possible split points.
        unique_values = np.unique(dataset[:, attribute_index])
        
        for value in unique_values:
            s_left, s_right = split_dataset(dataset, attribute_index, value)
            gain = information_gain(dataset, s_left, s_right)
            
            if gain > highest_gain:
                highest_gain = gain
                best_attribute_index = attribute_index
                best_value = value

    return best_attribute_index, best_value, highest_gain

def decision_tree_learning(dataset, max_depth=None, depth=0):
    """
    Recursively constructs the decision tree.

    This function is called recursively, resulting in a nested dictionary 
    where parent nodes have keys containing the child nodes as values.
    
    Parameters:
    - dataset (np.ndarray): Dataset to analyze.
    - depth (int): Current depth of the tree, default is 0.

    Returns:
    - dict: A node in the decision tree.
    """
    room_numbers = dataset[:, -1]
    
    if max_depth is not None and depth >= max_depth:
        return {
            "leaf": True,
            "label": np.bincount(room_numbers.astype('int')).argmax(),
            "left": None,
            "right": None,
            "depth": depth,
            "useful_decision_node": False
        }

    # Base case: If the dataset is pure or the dataset is empty.
    if len(np.unique(room_numbers)) <= 1:
        return {
            "leaf": True,
            "label": dataset[0, -1] if len(dataset) != 0 else None,
            "left": None,
            "right": None,
            "depth": depth,
            "useful_decision_node": False
        }

    # Find split point for the new decision tree node.
    best_attribute_index, best_value, highest_gain = find_split(dataset)
    s_left, s_right = split_dataset(dataset, best_attribute_index, best_value)

    return {
        "leaf": False,
        "attribute": best_attribute_index,
        "value": best_value,
        "left": decision_tree_learning(s_left, 
                                       max_depth=max_depth, 
                                       depth=depth+1),
        "right": decision_tree_learning(s_right, 
                                        max_depth=max_depth, 
                                        depth=depth+1),
        "depth": depth,
        "useful_decision_node": False,
        "precalculated_label": np.bincount(room_numbers.astype('int')).argmax()
    }
