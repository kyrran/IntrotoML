# -*- coding: utf-8 -*-

import numpy as np


def evaluate(test_db, decision_tree):
    """
    Return the accuracy of a single tree.

    Parameters:
    - test_db (np.array): test dataset (n x 8); 7 columns with measurements
                          from each Wifi emitter; 1 column with true labels
    - decision_tree (dict): tree from training data
        
    Returns:
    - accuracy (float): accuracy of decision_tree
    """
    predicted_label_array = predict_array(test_db, decision_tree)
    true_label = test_db[:, -1]
    
    return accuracy(true_label, predicted_label_array)


def predict_array(dataset, decision_tree):
    """
    Predict label of data point using the tree.

    Parameters:
    - decision_tree (dict): tree from training data
    - dataset (np.array): dataset (n x 8); 7 columns with 7 measurements
                          from each Wifi emitter; 1 column with labels
                          
    Returns:
    - prediction_array (np.array of int): predicted room number
    """
    dataset_x = dataset[:, :-1]
    prediction_array = [predict_single_value(row, decision_tree) 
                        for row in dataset_x]
    return np.array(prediction_array, dtype="int")


def predict_single_value(row, decision_tree):
    """
    Predict label of a row using the tree.

    Parameters:
    - decision_tree (dict): tree from training data
    - row (np.array): data (1 x 7); 7 columns with 7 measurements
                      from each Wifi emitter
                      
    Returns:
    - predicted_label (int): predicted room number for row
    """
    while not decision_tree["leaf"]:
        wifi_emitter_index = decision_tree["attribute"]
        corresponding_value = row[wifi_emitter_index]
        decision_tree = (decision_tree["left"] if corresponding_value <= 
                         decision_tree["value"] else decision_tree["right"])
            
    return decision_tree["label"]


def compute_conf_matrix(actual_labels, predicted_labels, class_labels=None):
    """
    Returns the normalised confusion matrix from actual and predicted labels.

    Parameters:
    - actual_labels(np.array): array of actual labels 
    - predicted_labels(np.array): array of predicted labels
    
    Returns:
    - confusion_matrix (2d Numpy array): for every unique class labels
    """
    if not class_labels:
        class_labels = np.unique(np.concatenate(
            (actual_labels, predicted_labels)))

    confusion_matrix = np.zeros(
        (len(class_labels), len(class_labels)), dtype="int")

    for i, label in enumerate(class_labels):
        indices = (actual_labels == label)
        predictions = predicted_labels[indices]
        unique_labels, counts = np.unique(predictions, return_counts=True)
        frequency_dict = dict(zip(unique_labels, counts))
        for j, class_label in enumerate(class_labels):
            confusion_matrix[i, j] = frequency_dict.get(class_label, 0)
            
    return confusion_matrix


def compute_norm_conf_matrix(confusion_matrix):
    row_sums = np.sum(confusion_matrix, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return confusion_matrix / row_sums


def accuracy(actual_labels, predicted_labels):
    """
    Returns accuracy from actual and predicted labels.

    Parameters:
    - actual_labels(np.array): array of actual labels 
    - predicted_labels(np.array): array of predicted labels
    
    Returns:
    - Accuracy(float): average accuracy value
    """
    assert len(actual_labels) == len(predicted_labels)
    try:
        return np.sum(actual_labels == predicted_labels) / len(actual_labels)
    except ZeroDivisionError:
        return 0.


def accuracy_from_confusion(confusion):
    """
    Compute the accuracy given the confusion matrix.
    
    Parameters:
    - confusion (np.ndarray): shape (C, C), where C is the number of classes.
                    Rows are ground truth per class, columns are predictions
                    
    Returns:
    - float : the accuracy
    """
    # Check if the total count in confusion matrix is non-zero
    if np.sum(confusion) > 0:
        # Return ratio of correctly predicted to the total predictions
        return np.sum(np.diag(confusion)) / np.sum(confusion)
    else:
        return 0.


def precision(confusion_matrix):
    """
    Compute the precision for each class and the macro-averaged precision
    from the confusion matrix.
    
    Parameters:
    - confusion_matrix (np.ndarray): Confusion matrix of predictions.
        
    Returns:
    - tuple: (Precision for each class, Macro-averaged precision)
    """
    p = np.zeros((len(confusion_matrix),))
    # Iterate through each class
    for c in range(confusion_matrix.shape[0]):
        col_sum = np.sum(confusion_matrix[:, c])
        # Avoid division by zero
        if col_sum > 0:
            p[c] = confusion_matrix[c, c] / col_sum

    # Calculate macro-averaged precision
    macro_p = np.mean(p) if len(p) > 0 else 0.
    return (p, macro_p)


def recall(confusion_matrix):
    """
    Compute the recall for each class and the macro-averaged recall 
    from the confusion matrix.
    
    Parameters:
    - confusion_matrix (np.ndarray): Confusion matrix of predictions.
        
    Returns:
    - tuple: (Recall for each class, Macro-averaged recall)
    """
    r = np.zeros((len(confusion_matrix),))
    for c in range(confusion_matrix.shape[0]):
        row_sum = np.sum(confusion_matrix[c, :])
        # Avoid division by zero
        if row_sum > 0:
            r[c] = confusion_matrix[c, c] / row_sum
            
    macro_r = np.mean(r) if len(r) > 0 else 0.
    return (r, macro_r)


def f1_score(confusion_matrix):
    """
    Compute the F1 score for each class and the macro-averaged F1 score
    from the confusion matrix.
    
    Parameters:
    - confusion_matrix (np.ndarray): Confusion matrix of predictions.
        
    Returns:
    - tuple: (F1 score for each class, Macro-averaged F1 score)
    """
    # Fetch precision and recall for each class
    precisions, macro_p = precision(confusion_matrix)
    recalls, macro_r = recall(confusion_matrix)
    assert len(precisions) == len(recalls)
    f = np.zeros((len(precisions),))
    for c, (p, r) in enumerate(zip(precisions, recalls)):
        # Calculate F1 score for each class
        f[c] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.

    # Calculate macro-averaged F1 score
    macro_f = np.mean(f) if len(f) > 0 else 0.
    return (f, macro_f)



def print_metrics_for_conf_matrix(conf_matrix, average_depth_tree):
    """
    Prints various classification metrics computed from a confusion matrix 
    and the average depth of a decision tree.

    Parmeters:
    - conf_matrix (np.ndarray): Confusion matrix where rows represent the 
                                actual classes and columns represent the 
                                predicted classes.
    - average_depth_tree (float): Average depth of the decision tree.

    Outputs:
    - No return value. Metrics are printed directly to the console.
    """
    accuracy = accuracy_from_confusion(conf_matrix)
    precision_per_class, precision_macro = precision(conf_matrix)
    recall_per_class, recall_macro = recall(conf_matrix)
    f1_score_per_class, f1_score_macro = f1_score(conf_matrix)
    
    print("The average confusion matrix is: ")
    print(conf_matrix)
    print("")
    print("From this average confusion matrix we can compute")
    print("the following average values:")
    print("")
    print("1) The accuracy is:")
    print(f"   {accuracy}.")
    print("")
    print("2) The Precision per class is: ")
    print(f"   {precision_per_class}")
    print("")
    print("3) The Macro-Precision is:")
    print(f"   {precision_macro}")
    print("")
    print("4) The Recall per class is:")
    print(f"   {recall_per_class}")
    print("")
    print("5) The Macro-Recall is:")
    print(f"   {recall_macro}")
    print("")
    print("6) The F1 Score per class is: ")
    print(f"   {f1_score_per_class}")
    print("")
    print("7) The Macro-F1-Score is:")
    print(f"   {f1_score_macro}")
    print("")
    print("8) The average depth of the decision tree is:")
    print(f"   {average_depth_tree}")
    print("")
    print("")
