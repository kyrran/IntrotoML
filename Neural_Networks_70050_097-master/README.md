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


### part2_house_value_regression.py

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
