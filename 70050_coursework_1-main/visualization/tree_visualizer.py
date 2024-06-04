# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from decision_tree.leaf_counter import count_widest_under, tree_depth


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

def plot_tree(node, ax):
    """
    Recursive function to plot a tree.

    Parameters:
    - node: The current node in the decision tree to plot.
    - x, y: The coordinates for the current node on the plot.
    - ax: The axis object from Matplotlib where the nodes and edges will be plotted.

    This function uses a depth-first approach to traverse and plot the tree. 
    """

    # If it's an internal node, display it with a blue box
    node_text = "test"
    # node_text = f"Attr: {node['attribute']} < {node['value']}"

    # Calulate relative position using widest_under
    nodes_to_left = count_decision_nodes_under(node["left"])
    nodes_to_right = count_decision_nodes_under(node["right"])
    
    x = nodes_to_left * 12 + 1
    y = node["depth"] * 8 + 1
    
    rect_width = 10
    rect_height = 3
    
    # Creating a rectangle
    rect = patches.Rectangle((x, y), width = 10 , height = 3, linewidth=1, facecolor='none')
    ax.add_patch(rect)
    
    # Adding text inside the rectangle (centered)
    text_x = x + rect_width / 2
    text_y = y + rect_height / 2
    
    plt.text(text_x, text_y, 'Sample Text', ha='center', va='center')

    # If the current node has a left child, draw a line to the left child and 
    # recursively plot the left subtree
    if not node["left"] is None:
        if node["left"]["leaf"] is False:
            plot_tree(node["left"], ax)
    else:
        pass
    

    # Similarly, if the current node has a right child, draw a line to the right child and 
    # recursively plot the right subtree
    # if node["right"]:
    #     ax.plot([x, right_x], [y, y - 1], 'k-')  # drawing a line to the child
    #     plot_tree(node["right"], right_x, y - 1, ax)
    
    # If the node is a leaf, it will be displayed with a green box
    # if node["leaf"]:
    #     leaf_text = f"R{node["label"]}"
    #     x = 
    #     y = depth * 7
    #     ax.text(x, y, leaf_text,
    #             bbox=dict(facecolor='#86E059'), ha='center')
    return

def visualize_tree(root):
    """
    Create a visualization of the decision tree.

    Parameters:
    - root: The root node of the decision tree.

    This function initializes a plot proportional to the tree's depth, and it calls the 
    recursive `plot_tree()` function to display the entire tree.
    """

    # Calculate dimensions of the tree for setting up the figure size
    n_decision_nodes = count_decision_nodes_under(root)
    width = n_decision_nodes * 12
    depth = tree_depth(root) * 8
    print(width)
    print(depth)

    # Initialize a figure object with a size that scales with the depth of the tree
    fig, ax = plt.subplots(figsize=(width, depth))
    
    # Call the recursive tree plotting function starting with the root
    plot_tree(root, ax=ax)
    
    ax.set_xlim(0, width) 
    ax.set_ylim(-depth, 0)
    # ax.set_aspect('equal', 'box')
    # Remove axis since they are irrelevant for a tree diagram
    # ax.axis('on')
    
    # Display the tree
    plt.show()