# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from decision_tree.leaf_counter import count_widest_under, tree_depth

def plot_tree(node, x, y, ax):
    """
    Recursive function to plot a tree.

    Parameters:
    - node: The current node in the decision tree to plot.
    - x, y: The coordinates for the current node on the plot.
    - ax: The axis object from Matplotlib where the nodes and edges will be plotted.

    This function uses a depth-first approach to traverse and plot the tree. 
    """
    # If the node is a leaf, it will be displayed with a green box
    if node["leaf"]:
        leaf_text = ("R{:n}".format(node["label"]) if node["label"] else "Empty")
        ax.text(x, y, leaf_text,
                bbox=dict(facecolor='#86E059'), ha='center')
        return

    # If it's an internal node, display it with a blue box
    node_text = "Attr {} > {:.0f}".format(node["attribute"], node["value"])
    ax.text(x, y, node_text, bbox=dict(facecolor='#B1B4D3'), ha='center')

    # Calulate relative position using widest_under
    span_left = 0
    span_right = 0

    if node["left"]:
        span_right = 1 if node["left"]["leaf"] else node["right"]["widest_under"]
        span_left = 1 if node["right"]["leaf"] else node["right"]["widest_under"]

    left_x = x - span_right / 2
    right_x = x + span_left / 2

    # If the current node has a left child, draw a line to the left child and 
    # recursively plot the left subtree
    if node["left"]:
        ax.plot([x, left_x], [y, y - 1], 'k-')  # drawing a line to the child
        plot_tree(node["left"], left_x, y - 1, ax)

    # Similarly, if the current node has a right child, draw a line to the right child and 
    # recursively plot the right subtree
    if node["right"]:
        ax.plot([x, right_x], [y, y - 1], 'k-')  # drawing a line to the child
        plot_tree(node["right"], right_x, y - 1, ax)

def visualize_tree(root):
    """
    Create a visualization of the decision tree.

    Parameters:
    - root: The root node of the decision tree.

    This function initializes a plot proportional to the tree's depth, and it calls the 
    recursive `plot_tree()` function to display the entire tree.
    """
    # Count and record the widest possible depth under each node of the tree
    width_list = count_widest_under(root)


    # Calculate dimensions of the tree for setting up the figure size
    width = root["widest_under"]
    depth = tree_depth(root)

    # Initialize a figure object with a size that scales with the depth of the tree
    fig, ax = plt.subplots(figsize=(1.5*width, 1.5*depth))
    
    # Call the recursive tree plotting function starting with the root
    plot_tree(root, x=0, y=0, ax=ax)
    
    # Remove axis since they are irrelevant for a tree diagram
    ax.axis('off')
    
    # Display the tree
    plt.show()