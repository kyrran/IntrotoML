# -*- coding: utf-8 -*-

from copy import deepcopy
from training_and_evaluation.evaluation_utilities import evaluate

def pruning_single_tree(decision_tree, val_data):
    """
    Prunes given tree based on validation data and returns original tree.

    Parameters:
        - decision_tree (dict): Tree to be pruned.
        - val_data (np.ndarray): Validation data for pruning.

    Returns:
        - dict: Original, unpruned decision tree.
    """
    # Backup original tree for return
    backup_tree = deepcopy(decision_tree)

    # Perform pruning on the decision_tree
    try_prune_node(decision_tree, decision_tree, val_data)

    return backup_tree

def try_prune_node(decision_tree, node, val_data):
    """
    Recursively prunes tree nodes, starting from the given node.

    Parameters:
        - decision_tree (dict): Complete decision tree.
        - node (dict): Current node considered for pruning.
        - val_data (np.ndarray): Validation data for pruning.
    """
    
    # Return if node is a leaf (base case)
    if node["leaf"]:
        return

    # Recursively attempt to prune child nodes
    if node["left"]:
        try_prune_node(decision_tree, node["left"], val_data)
    if node["right"]:
        try_prune_node(decision_tree, node["right"], val_data)

    # Check if either child is marked as useful
    if (node["left"]["useful_decision_node"] or 
        node["right"]["useful_decision_node"]):
        node["useful_decision_node"] = True
        return

    # Prune node if both children are leaves
    if node["left"]["leaf"] and node["right"]["leaf"]:
        original_tree = deepcopy(decision_tree)
        backup_left, backup_right = node["left"], node["right"]

        # Construct the pruned node
        node.update({
            "leaf": True, 
            "label": node["precalculated_label"], 
            "left": None, 
            "right": None
        })

        # Check performance of pruned vs original
        if pruned_outperformes_original(decision_tree, 
                                        original_tree, 
                                        val_data):
            return
        else:
            # Restore original node if no improvement
            node.update({
                "useful_decision_node": True, 
                "leaf": False, 
                "label": None, 
                "left": backup_left, 
                "right": backup_right
            })
            return

def pruned_outperformes_original(pruned_tree, original_tree, val_data):
    """
    Compares performance of pruned and original trees with validation data.

    Parameters:
        - pruned_tree (dict): Pruned decision tree.
        - original_tree (dict): Original decision tree.
        - val_data (np.ndarray): Validation data for comparison.

    Returns:
        - bool: True if pruned tree performs better or equally; else False.
    """
    pruned_accuracy = evaluate(val_data, pruned_tree)
    original_accuracy = evaluate(val_data, original_tree)

    return pruned_accuracy >= original_accuracy
