# -*- coding: utf-8 -*-
# Main file

import sys

from utilities.dataset_loader import load_dataset
from task_3 import task_3
from task_4 import task_4
from decision_tree.decision_tree import decision_tree_learning
from visualization.tree_visualizer1 import visualize_tree
    
def main():
    """
    Main function to execute tasks 3 and 4 for both clean and noisy datasets.
    
    This function:
    1. Loads the clean and noisy datasets.
    2. Executes and prints results for task 3 for both datasets.
    3. Executes and prints results for task 4 for both datasets.
    
    Returns:
    - No return value. Results are printed directly to the console.
    """

    # Load the clean dataset
    dataset_clean = load_dataset("data/clean_dataset.txt")

    # Load the noisy dataset too
    dataset_noisy = load_dataset("data/noisy_dataset.txt")

    run_task3(dataset_clean, dataset_noisy)
    run_task4(dataset_clean, dataset_noisy)

    return

def run_task3(dataset_clean = None, dataset_noisy = None):
    """
    Execute task 3 for both clean and noisy datasets.

    This function:
    1. Loads the clean and noisy datasets if not provided.
    2. Executes and prints results for task 3 for both datasets.

    Returns:
    - No return value. Results are printed directly to the console.
    """
    if dataset_clean is None:
        dataset_clean = load_dataset("data/clean_dataset.txt")
    if dataset_noisy is None:
        dataset_noisy = load_dataset("data/noisy_dataset.txt")

    # Task 3
    print("###########################################")
    print("#### Task 3 ###############################")
    print("###########################################")
    print("")
    print("")
    print("For the clean dataset we get the following results:")
    print("")
    task_3(dataset_clean)
    print("")
    print("")
    print("For the noisy dataset we get the following results:")
    print("")
    task_3(dataset_noisy)

    return

def run_task4(dataset_clean = None, dataset_noisy = None):
    """
    Execute task 4 for both clean and noisy datasets.

    This function:
    1. Loads the clean and noisy datasets if not provided.
    2. Executes and prints results for task 3 for both datasets.

    Returns:
    - No return value. Results are printed directly to the console.
    """

    if dataset_clean is None:
        dataset_clean = load_dataset("data/clean_dataset.txt")
    if dataset_noisy is None:
        dataset_noisy = load_dataset("data/noisy_dataset.txt")

    # Task 4
    print("###########################################")
    print("#### Task 4 ###############################")
    print("###########################################")
    print("")
    print("")
    print("For the clean dataset we get the following results:")
    print("")
    task_4(dataset_clean)
    print("")
    print("")
    print("For the noisy dataset we get the following results:")
    print("")
    task_4(dataset_noisy)

    return

def tree_gen(dataset_path = None):
    """
    Use decicison_tree_learning() to generate the decision tree of a dataset.
    Visualise it with matplotlib.

    This function:
    1. Loads the dataset if not provided.
    2. Learn from the dataset and visualise the result tree.

    Returns:
    - No return value. Results are printed directly to the console.
    """
    if dataset_path is None:
        dataset_path = "data/clean_dataset.txt"

    dataset = load_dataset(dataset_path)
    tree = decision_tree_learning(dataset)
    visualize_tree(tree)


if __name__ == "__main__":
    params = sys.argv[1:]

    if len(params) == 0:
        main()
        exit()

    match params[0]:
        case "-help":
            print("Params:\n"
                  "  -task3 :Run task3 with default dataset.\n" +
                  "  -task4 :Run task4 with default dataset.\n" +
                  "  -tree [datafile_path] :Train and visualise the tree with given dataset\n" +
                  "                         Use clean_dataset if no path provided")
        case "-task3":
            run_task3()
        case "-task4":
            run_task4()
        case "-tree":
            if len(params) > 1:
                tree_gen(params[1])
            else:
                tree_gen()
        case _:
            main()
    exit()
    
