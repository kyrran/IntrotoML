# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def count_decision_nodes_under(node):
    """
    Counts the number of decision nodes under a given node.
    
    Parameters:
    - node (dict): The current node.
    
    Returns:
    - int: Number of decision nodes under the node.
    """
    # Base case: if node is None or it's a leaf, return 0
    if node is None or node["leaf"]:
        return 0

    # Recursively get the count for left and right children
    left_count = count_decision_nodes_under(node["left"])
    right_count = count_decision_nodes_under(node["right"])

    # Add one for the current decision node and the counts from left and right subtrees
    return 1 + left_count + right_count

def plot_node(ax, x, y, text, node_type):
    """Helper function to plot a node."""
    if node_type == "decision":
        rect = patches.Rectangle((x, y), 10, 5, facecolor='grey')
        ax.add_patch(rect)
        ax.text(x + 5, y + 2.5, text, ha='center', va='center')
    else:
        rect = patches.Rectangle((x, y), 5, 5, facecolor='green')
        ax.add_patch(rect)
        ax.text(x + 2.5, y + 2.5, text, ha='center', va='center')

def plot_tree(tree, ax, x=0, y=0, x_step=15):
    """Recursive function to plot the tree."""
    if tree['leaf']:
        plot_node(ax, x, y, str(tree['label']), 'leaf')
    else:
        plot_node(ax, x, y, f"A:{tree['attribute']} V:{tree['value']}", 'decision')
        
        # Plot the left and right branches recursively
        if tree['left']:
            plot_tree(tree['left'], ax, x - x_step / (2**tree['depth']), y - 10)
        if tree['right']:
            plot_tree(tree['right'], ax, x + x_step / (2**tree['depth']), y - 10)

def visualize_tree(tree):
    fig, ax = plt.subplots(figsize=(15, 10))
    plot_tree(tree, ax)
    ax.axis('off')
    plt.show()

visualize_tree(tree)