# Intro2ML_23/Coursework1

## Introduction 

Repository for *COMP70050 Introduction to Machine Learning* coursework 1.

Decision Tree Classifier for Indoor Location Identification

This project implements a decision tree classifier to determine indoor locations based on WIFI signal strengths collected from a mobile phone.

## Instructions

The program entry is `main.py`. It can be invoked with the defaut `python3 main.py`(Linux) or `py main.py`(Windows) command.
In this case ite executes both tasks 3 and 4, printing all necessary metrics.

The program also takes command line arguments to be executed alternatively:
- `-help` Print this help prompt.
- `-task3` Run task3 only, with default datasets.
- `-task4` Run task4 only, with default datasets.
- `-tree [dataset_path]` Train and visualise the tree with provided dataset. Use clean_dataset if no path provided.


## Three important scripts:

    main.py: (RUN THIS FUNCTION FOR TESTING THE ENTIRE PROBLEM)
        Purpose: The main driver script that combines all modules.
        Functions/Procedures:
            Loads datasets, both clean and noisy dataset
            Executes task 3 for both data
            Executes task 4 for both data

    task_3.py:
        Purpose: To do Evaluation and calculate evaluation and performance matrices.
        Functions/Procedures:
            Splits the data into 10 fold (using split_dataset_into_k_folds)
            Train and return k decision trees and k test_data (using k_fold_training)
            Calculate tree depth and average depth (using tree_depth)
            Combines 10 normalized confusion matrices into 1 (using av_confusion_matrix)
            Prints confusion matrix and other evaluation parameters (using print_metrics_for_conf_matrix)

    task_4.py:
        Purpose : To do nested cross validation pruning and display confusion matrix and evaluation parameters.
        Functions/Procedures:
            Performs cross validation on the 10 folds generated in Task 3 (using nested_cross_validation_pruning) 
            Normalizes all conf_matrices in og_results and pruned_results (using compute_norm_conf_matrix)
            Prints metrics for confusion matrix (using print_metrics_for_conf_matrix) 

## Directory Structure:

    |-- main.py
    |-- task_3.py
    |-- task_4.py
    |-- decision_tree
        |-- decision_tree.py
        |-- information_gain.py
        |-- leaf_counter.py
    |-- pruning
        |-- ncv_pruning.py
        |-- pruning.py
    |-- training_and_evaluation
        |-- evaluation_utilities.py
        |-- hyperparameter_tuning.py
        |-- k_fold_splitting.py
        |-- k_fold_training.py
        |-- step3.py
    |-- utilities
        |-- dataset_loader.py
    |-- visualization
        |-- tree_visualizer.py
    |-- data
        |-- clean dataset.txt
        |-- noisy dataset.txt
    |-- docs
        |-- Coursework Setup Guide.pdf
        |-- spec.pdf


## Modules Description

    decision_tree:

        decision_tree.py:
            Purpose: Contains the main decision tree learning algorithm.
            Functions:
                Identifies the best attribute and value for splitting based on maximal information gain (using find_split)
                Recursively constructs the decision tree(using decision_tree_learning)
    
        information_gain.py:
            Purpose: Provides functions for computing information gain, entropy, and remainder for datasets.
            Functions:
                Computes the dataset's entropy (using entropy)
                Calculates the remainder for subsets Sleft and Sright (using remainder)
                Computes information gain (using information_gain)
        
        leaf_counter.py:
            Purpose: Counts the leafs under a node
            Functions: 
                Compute the depth of a tree rooted at the given node. (using tree_depth)
    
    pruning:

        ncv_pruning.py:
            Purpose: Performs nested cross validation pruning
            Functions:
                Performs nested cross validation pruning (using nested_cross_validation_pruning)

        pruning.py:
            Purpose: Performs traversing to find pruning and perform single tree pruning.
            Functions:
                Prune decision_tree to a modified, pruned version using val_data (using pruning_single_tree)
                Recursively traverse the tree to find pruning opportunity (using try_prune_node)
                Evaluates the two trees' performance (using pruned_outperformes_original)
    
    training_and_evaluation:

        evaluation_utilities.py:

            Purpose: Contains all the utilities for evaluation like normalised Confusion matrix, accuracy, precision and f1 score
            Functions:  
                Return the accuracy value of a single tree (using evaluate)
                Predict the label of the data point using the decision tree (using predict_array)
                Predict the label of the row in the dataset using the decision tree(using predict_single_value)
                Returns the confusion matrix,accuract,precision,recall,f1(using compute_conf_matrix,accuracy,precision,recall,f1 score)
                Prints metrics for confusion_matrix (using print_metrics_for_conf_matrix)

        
        hyperparameter_tuning.py:
            Purpose: Train k models on the k training sets and return optimal hyperparameter
            Functions:
                Train k models on trainig sets with maximum depth possible (using hyperparameter_tuning).
                Returns maximum depth among all of the k models(using find_max_depth)
        
        k_fold_splitting.py:
            Purpose: Splits the data into k folds
            Functions:
                Split n_instances into n mutually exclusive splits at random (using k_fold_split)
                Generate train and test indices at each fold (using split_dataset_into_k_folds)

        k_fold_training.py:
            Purpose: Train k models on the k training sets and return all models



    utilities:

        dataset_loader.py:
            Purpose: Handles data loading and preprocessing tasks.
            Functions:
                Loads a dataset from the specified filename (using load_dataset)
                Splits the dataset based on a specified attribute and value(using split_dataset)

    visualization:

        tree_visualizer.py:
            Purpose: Offers visualization tools for the constructed decision tree.
            Functions:
                Provides a basic visualization of the decision tree using matplotlib. (using plot_tree)
                Create a visualization of the decision tree. (using visualize_tree)


## Dependencies:

This project relies on:

    numpy
    matplotlib
    Standard Python libraries: Python Standard Library

-------------
------------
-----------

# Intro2ML_23/Coursework2

Repository for *COMP70050 Introduction to Machine Learning* coursework 2: Neural Networks.

## Requirements

- pandas
- torch
- pickle
- numpy
- sklearn

For automated hyperparameter tuning:

- skorch

> **Note**: skorch is not available in the lab environment, but its use is limited to hyperparameter tuning and does not affect the testing.

## Instructions

### Part 2

To replicate our training process using the conditions we used for submission:

```lang=bash
py part2_house_value_regression.py (Windows)
python3 part2_house_value_regression.py (Linux/MacOS)
```

To evaluate the final model, change the function in part2_house_value_regression.py (line: 657) from `training_main()` to `final_eval()`:

```lang=python
if __name__ == "__main__":
    training_main()
```

This will instead print the final evaluation score of model stored in `part2_model.pickle`.

## Code Structure

### part1_nn_lib.py

This file is a custom neural network library and includes foundational components for building and training neural networks. The main elements are:

#### Layer Classes:

- `Layer`: An abstract base class for different types of layers.
- `LinearLayer`: Implements affine transformation (Wx + b).
- `SigmoidLayer` and `ReluLayer`: Provide sigmoid and ReLU activation functions, respectively.

#### Loss Function Classes:

- `MSELossLayer`: Calculates mean-squared error, used for regression tasks.
- `CrossEntropyLossLayer`: Implements cross-entropy loss, typically used for classification tasks.

#### Network and Training Utilities:

- `MultiLayerNetwork`: A flexible class for creating a neural network with customizable layers and activations.
- `Trainer`: Manages the training process, including forward and backward passes, and parameter updates.

#### Data Preprocessing:

- `Preprocessor`: Normalizes data and provides a method to revert normalization, crucial for processing real-world datasets.

#### Utility Functions:

- `save_network` and `load_network`: Functions for saving and loading the trained model, useful for long training processes or model deployment.
- `xavier_init`: Implements Xavier initialization for network weights, aiding in the convergence of training.

#### Example Usage

`part1_nn_lib.py` includes an example function `example_main`, demonstrating the creation of a network, data preprocessing, and training with the Trainer class. This example uses the Iris dataset and can be run to see a simple demonstration of how the library components work together.


#### part2_house_value_regression.py

---

`CustomNet`

- `__init__`: Construct a custom network with parameters
- `forward`: Custom forward pass based on the network structure

`Regressor`

- `__init__`: Construct the regressor, build network, fill the preprocessor values
- `_preprocessor`: Preprocess data in both training and evaluation mode
- `fit`: Train the model with data
- `predict`: Use current network to predict value with input
- `score`: Get a performance score of current network from given data

Other functions:
- `save_regressor`: Save the whole regressor into a pickle file
- `load_regressor`: Load the regressor from the default pickle file
- `save_loss_data`: Save the loss during training to a csv file
- `save_eval_data`: Save the evaluation loss during training to a csv file
- `rmse`: Calculate RMSE loss
- `log_rmse`: Calculate Log_RMSE loss
- `RegressorHyperParameterSearch`: Perform an automated hyperparameter tuning
- `training_main`: A complete procedure of the model training
- `transfer_trained_to_cpu`: Transfer the model back to CPU if it was trained with GPU
- `final_eval`: Get a deterministic evaluation score for the saved model
