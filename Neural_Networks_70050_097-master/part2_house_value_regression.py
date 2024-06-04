import pandas
import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV
#from skorch import NeuralNetRegressor
import torch.nn.functional as F
import torch.utils.data as u_data
from datetime import datetime

class CustomNet(nn.Module):
    """
    Custom torch module for better customisation on network layers
    """
    def __init__(self,
                 input_size,
                 hidden_layer_number = 1,
                 hidden_layer_feature = 64,
                 activation_function = nn.ReLU,
                 init_function = None,
                 apply_dropout = False,
                 dropout_rate = 0.2
                 ):
        """
        Construct the network and it's linear and activation layers

        Args:
            input_size {int}: Number of attributes of input data
            hidden_layer_number {int}: number of hidden layers of the network
            hidden_layer_feature {int}: number of neurons of each hidden layer
            activation_function {function}: type of activation function to use
            init_function {function}: type of parameter initialization function to use
            apply_dropout {boolean}: apply dropout after each forward pass or not
            dropout_rate {float}: dropout rate if apply dropout
        """
        super().__init__()

        self.apply_dropout = apply_dropout

        # Use ModuleList allowing the network to be transferred to GPU
        # so I can train it with 500k epoch
        self.layers = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Add input layer
        self.layers.append(nn.Linear(input_size, hidden_layer_feature))
        self.acts.append(activation_function())
        self.dropouts.append(nn.Dropout(dropout_rate))

        # Add hidden layers (if more than 1)
        if hidden_layer_number > 1:
            for i in range(hidden_layer_number - 1):
                self.layers.append(nn.Linear(hidden_layer_feature, hidden_layer_feature))
                self.acts.append(activation_function())
                self.dropouts.append(nn.Dropout(dropout_rate))

        # Add output layer
        self.output = nn.Linear(hidden_layer_feature, 1)

        # If init function provided, fill in the init values
        if init_function:
            for layer in self.layers:
                init_function(layer.weight)
            init_function(self.output.weight)

    def forward(self, x):
        """
        Apply a forward pass to given data x

        Args:
            x {torch.tensor}: training data

        Returns:
            predictions from this pass
        """
        for layer, act, dropout in zip(self.layers, self.acts, self.dropouts):
            x = act(layer(x))
            # Apply dropout if we want to
            if self.apply_dropout:
                x = dropout(x)
        x = self.output(x)
        return x

class Regressor():

    def __init__(self, x, nb_epoch = 1000, apply_dropout = False, dropout_rate = 0.1):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.
            - apply_dropout {float} -- apply dropout after each forward pass or not
            - dropout_rate {float} -- dropout rate if apply dropout
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Prepare parameter dictionary for preprocessor
        self.pre_params = dict()

        # Fill the parameter dict and get input_size
        X, _ = self._preprocessor(x, training = True)

        # Fill other class fields
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch

        # Creating custom network with the parameters
        self.net = CustomNet(
            self.input_size,
            2,
            512,
            apply_dropout= apply_dropout,
            dropout_rate= dropout_rate)

        # Create a list to store loss over the training for plotting
        self.loss_data = []
        self.eval_data = []
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be
              the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Reset the index so the LabelBinarizer concats correctly
        x.reset_index(drop= True, inplace=True)

        # Split numerical and categorical for processing
        numerical = x.select_dtypes(include=['float64', 'int64'])
        categorical = x.select_dtypes(include=['object', 'category'])

        # Fill the parameters dict to use in future non-training processing
        if training:
            self.pre_params['numerical_median'] = numerical.median()
            self.pre_params['categorical_most'] = []
            for column in categorical.columns:
                self.pre_params['categorical_most'].append(x[column].mode()[0])
            self.pre_params['numerical_min'] = numerical.min()
            self.pre_params['numerical_max'] = numerical.max()


        # filling missing numerical columns with median value
        numerical = numerical.fillna(self.pre_params['numerical_median'])

        # filling missing text columns with the most frequent value
        for column_index in range(len(categorical.columns)):
            column = categorical.columns[column_index]
            categorical[column] = categorical[column].fillna(
                self.pre_params['categorical_most'][column_index]
            )

        # binarize text column
        lb = LabelBinarizer()
        categorical_new = pd.DataFrame()
        for column in categorical.columns:
            # combines fit and transform into a single step
            instance = lb.fit_transform(categorical[column])
            # covert from numpy to dataframe
            instance_df = pd.DataFrame(instance, columns=lb.classes_)
            # drop the original column
            categorical_new = pd.concat([categorical_new, instance_df], axis=1)

        # Normalize numerical value
        min_value = self.pre_params['numerical_min']
        max_value = self.pre_params['numerical_max']
        numerical = (numerical - min_value) / (max_value - min_value)

        # Concat the two parts back to data
        x = pd.concat([numerical, categorical_new], axis=1)

        # Convert to tensor
        x = torch.tensor(x.values, dtype=torch.float32)
        if y is None:
            return x, None
        y = torch.tensor(y.values, dtype=torch.float32)
        # Return preprocessed x and y, return None for y if it was None
        return x, y
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y, x_eval = None, y_eval = None):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # All mannual-set parameters here for easy modifying
        batch_size = 64
        learning_rate = 1

        # Preprocess
        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget


        # !!!!!!!!! PICK ONE OF THE FOLLOWING
        # !!! USING GPU
        #X = X.cuda()
        #Y = Y.cuda()
        #net = self.net.cuda()

        # !!! USING CPU
        net = self.net

        # use torch.utils to batch data
        dataset = u_data.TensorDataset(X, Y)
        training_loader = u_data.DataLoader(dataset, batch_size = batch_size, shuffle=True)

        # choose loss function
        loss = nn.MSELoss()

        # choose optimiser
        optimiser = torch.optim.Adadelta(
            net.parameters(),
            lr = learning_rate,
            weight_decay= 0
        )

        # Train all epochs
        for i in range(self.nb_epoch):
            print("Epoch {}:".format(i+1))
            running_loss = 0.
            last_loss = 0.

            # Train per epoch
            for batch_index, data in enumerate(training_loader):
                inputs, labels = data
                optimiser.zero_grad()
                pred_y = net(inputs)
                l = torch.sqrt(loss(pred_y, labels))
                l.backward()
                optimiser.step()

                # Print out the metrics
                running_loss += l.item()
                if batch_index % 10 == 9:
                    last_loss = running_loss / 10
                    print("batch {} loss: {}".format(batch_index + 1, last_loss))
                    running_loss = 0

            # Keep record of loss/eval data for plots
            self.loss_data.append(last_loss)
            if not (x_eval is None):
                score_epoch = self.score(x_eval, y_eval)
                self.eval_data.append(score_epoch)
                print("Average Loss on validation of epoch: {}".format(score_epoch))
            print("Average Loss of epoch: {}".format(last_loss))
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, _ = self._preprocessor(x, training=False)  # Do not forget

        # !!!!!!!!!!! COMMENT THIS IF NOT USING GPU
        # !!! USING GPU
        #X = X.cuda()

        # disable gradient calculations.
        # In prediction mode, you don't need to compute gradients,
        # which are only necessary during training for backpropagation
        with torch.no_grad():
            predictions = self.net(X)

        # !!! AND THIS .cpu()
        #return predictions.cpu().numpy()
        return predictions.numpy()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        pred_y = self.predict(x)
        pred_y = torch.from_numpy(pred_y)
        Y = torch.tensor(y.values, dtype=torch.float32)
        return rmse(pred_y, Y)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model, label):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    With extra label to identify them

    Args:
        trained_model {Regressor}: trained model
        label {str}: 'final' if the model is used for submission,
                     anything else to add a label to identify the models
    """
    # If you alter this, make sure it works in tandem with load_regressor
    if label == 'final':
        label = ''
    else:
        label = '_' + label
    with open(f'part2_model{label}.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print(f"\nSaved model in part2_model{label}.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model

def save_loss_data(loss_data, time_str):
    """
    Utility function to save the loss data to a csv file with timestamp

    Args:
        loss_data {list}: List of loss during training per epoch
        time_str {string}: String of timestamp in the form of DD_HHMMSS
    """
    np.savetxt("loss_{}.csv".format(time_str),
               loss_data,
               delimiter=',',
               fmt='%f')
    print("\nSaved Loss data in loss_{}.csv".format(time_str))

def save_eval_data(eval_data, time_str):
    """
    Utility function to save the evaluation loss data to a csv file with timestamp

    Args:
        loss_data {list}: List of loss evaluated using an evaluation set
                          during training per epoch
        time_str {string}: String of timestamp in the form of DD_HHMMSS
    """
    np.savetxt("eval_{}.csv".format(time_str),
               eval_data,
               delimiter=',',
               fmt='%f')
    print("\nSaved Loss data in eval_{}.csv".format(time_str))


def rmse(pred_y, y):
    """
    Using rmse to get the error.

    Args:
        pred_y {torch.tensor}: Predicted label from the input
        y {torch.tensor}: True label of the input

    Returns:
        {float}: rmse score of this prediction (lower is better)
    """
    mse_loss = F.mse_loss(pred_y, y)

    # Calculate RMSE
    rmse = torch.sqrt(mse_loss)

    return rmse

def log_rmse(pred_y, y):
    """
    Use rmse for the logged value to get a better insight into relative error

    Args:
        pred_y {torch.tensor}: Predicted label from the input
        y {torch.tensor}: True label of the input

    Returns:
        {float}: log_rmse score of this prediction (lower is better)

    """
    # Log_rmse is currently broken so we're back to just rmse

    # Clamp the values to 1 to furthur stablise the result
    clamp_y = torch.clamp(pred_y, 1, float('inf'))

    # Calculating the log_rmse
    l = torch.sqrt(torch.mean((torch.log(clamp_y) - torch.log(y))**2))
    return l

def RegressorHyperParameterSearch(x, y):
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Lines using custom packages are commented out to enable the project
    to be tested on LabTS.

    Arguments:
        - x {pd.DataFrame} -- Raw input array of shape
            (batch_size, input_size).
        - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    # Create a dummy model so we can use the preprocessor
    dummy_regressor = Regressor(x)
    X, Y = dummy_regressor._preprocessor(x=x, y=y, training=True)

    # Convert our custom torch model to sklearn model using skorch
    #model = NeuralNetRegressor(
    #    module=CustomNet,
    #    optimizer= torch.optim.Adam,
    #    max_epochs= 300
    #)

    # Define our parameter table for small-scaled tuning automation
    parameters_to_tune = {
        # this is fixed
        'module__input_size': [X.shape[1]],
        'module__hidden_layer_number' : [1],
        #'module__hidden_layer_number' : [1, 2, 3],
        'module__hidden_layer_feature' : [512],
        #'module__hidden_layer_feature' : [64, 128, 256, 512, 1024],
        'module__activation_function': [nn.ReLU],
        #'module__activation_function' : [nn.ReLU, nn.ReLU6, nn.Sigmoid, nn.Tanh],
        'module__init_function': [nn.init.normal_],
        #'module__init_function' : [nn.init.uniform_,
        #                           nn.init.xavier_uniform_,
        #                           nn.init.normal_,
        #                           nn.init.zeros_,
        #                           nn.init.kaiming_normal_,
        #                           nn.init.kaiming_uniform_,
        #                           nn.init.xavier_normal_],
        'optimizer': [torch.optim.Adadelta],
        #'optimizer': [torch.optim.Adam,
        #              torch.optim.Adadelta,
        #              torch.optim.Adagrad,
        #              torch.optim.AdamW,
        #              torch.optim.Adamax,
        #              torch.optim.NAdam,
        #              torch.optim.RMSprop
        #              ],
        'optimizer__lr': [250],
        #'optimizer__lr': [100, 300, 500, 700],
        'optimizer__weight_decay': [0],
        'max_epochs': [1000],
        'module__apply_dropout': [True],
        #'module__apply_dropout': [True, False],
        'module__dropout_rate': [0.1],
        #'module__dropout_rate': [0.1, 0.2, 0.3, 0.4],
        'batch_size': [128]
        #'batch_size': [16, 32, 64, 128, 256]
    }

    # Perform the tuning
    # change n_jobs to less if the cpu is <16 cores
    #grid = GridSearchCV(
    #    estimator=model,
    #    param_grid=parameters_to_tune,
    #    scoring='neg_root_mean_squared_error',
    #    n_jobs=16,
    #    verbose=2
    #)
    #grid_result = grid.fit(X, Y)

    # Retrive the results
    #print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #means = grid_result.cv_results_['mean_test_score']
    #stds = grid_result.cv_results_['std_test_score']
    #params = grid_result.cv_results_['params']
    #for mean, stdev, param in zip(means, stds, params):
    #    print("%f (%f) with: %r" % (mean, stdev, param))

    return  #grid_result.best_params_ # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def training_main():
    """
    Primary function used to run the training and tuning

    """

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv")

    # Splitting training and evaluation
    data_train, data_evaluation = train_test_split(
        data,
        test_size=0.33,
        random_state=70050)

    # Splitting input and output
    x_train = data_train.loc[:, data.columns != output_label]
    y_train = data_train.loc[:, [output_label]]

    x_eval = data_evaluation.loc[:, data.columns != output_label]
    y_eval = data_evaluation.loc[:, [output_label]]

    # Create our model
    regressor = Regressor(
        x_train,
        nb_epoch = 5000,
        apply_dropout= True,
        dropout_rate= 0.1
    )

    # Training
    regressor.fit(x_train, y_train, x_eval, y_eval)

    # Save model and data
    save_regressor(regressor, 'final')
    time_str = datetime.now().strftime('%d_%H%M%S')
    save_loss_data(regressor.loss_data, time_str)
    save_eval_data(regressor.eval_data, time_str)

    # An intuitive view of performance
    x_pred = regressor.predict(x_eval)
    print(x_pred[0:10])
    print(y_eval[0:10])
    print(x_pred[-10:])
    print(y_eval[-10:])

    # Error
    error = regressor.score(x_eval, y_eval)
    print("\nRegressor error: {}\n".format(error))

    #RegressorHyperParameterSearch(x_train, y_train)

def transfer_trained_to_cpu():
    '''
    Transfer the net back to cpu so it can pass the LabTS tests
    '''
    my_model = load_regressor()
    my_model.net = my_model.net.to('cpu')
    save_regressor(my_model, 'final')

def final_eval():
    """
    Print the final evaluation score to use in report

    """
    output_label = "median_house_value"
    data = pd.read_csv("housing.csv")

    # Splitting training and evaluation
    data_train, data_evaluation = train_test_split(
        data,
        test_size=0.33,
        random_state=70050)

    x_eval = data_evaluation.loc[:, data.columns != output_label]
    y_eval = data_evaluation.loc[:, [output_label]]

    my_model = load_regressor()

    # disable the dropout layers so it will get the fixed result
    my_model.net.eval()

    final_score = my_model.score(x_eval, y_eval)
    print(final_score)

if __name__ == "__main__":
    training_main()

