# Random forest classifier - object oriented implementation

import numpy as np
import pandas as pd
from scipy import stats

from dataset import DataSet
from decision_tree import DecisionTree

class RandomForest():
    """
    Class for random forest classifier. Uses the DecisionTree class, all of the assumptions
    of the DecisionTree class are held for this classifier too (see docstring). The classifier
    uses the bagging algorithm in the fit stage, receiving bootstrap datasets and training a different
    tree for each dataset. The classifier also selects a random subset of the original features
    for each tree to train on. In the prediction stage, each test sample receives a prediction from
    each tree, and the mode value is selected as the final prediction. Attributes are assigned in the fit method.
    Attributes:
        bootstraps (list): Each value is a numpy array of m0*n, a bootstrap sample of the original train data
        num_features (int): The number of features to be sampled for each tree
        trees (list): Each value is a DecisionTree that is part of the forest
        tree_columns (list): Each value is a 1d numpy array of length num_features, containing the column
            indexes chosen for the i`th tree
        prune_value (int): the maximum depth that the trees are allowed to grow to
    """
    
    def __init__(self):
        self.bootstraps = None
        self.num_features = None
        self.trees = None
        self.tree_columns = None
        self.prune_value = None
    
    
    def fit(self, bootstraps, num_features, prune_value):
        """
        The training method, Chooses a random subset of <num_features> columns for each bootstrap,
        creates a tree and trains it on this bootstrap. The trees are saved in the <trees> attribute.
        The indexes of the chosen columns are saved in the <tree_columns> attribute.
        Params:
            bootstraps (list): Each value is a numpy array of m*n, a bootstrap sample of the original train data
            num_features (int): The number of features to be sampled for each tree
            prune_value (int): the maximum depth that the trees are allowed to grow to
        """
        self.bootstraps = bootstraps
        self.num_features = num_features
        self.prune_value = prune_value
        self.trees, self.tree_columns = [], []
        n = bootstraps[0].shape[1] - 1
        
        for i,strap in enumerate(bootstraps):
            columns = np.random.choice(np.arange(n), self.num_features, replace=False)
            columns = np.append(columns,-1)
            strap = strap[:,columns]
            t = DecisionTree()
            t.fit(strap, self.prune_value)
            self.trees.append(t)
            self.tree_columns.append(columns)
            
        
        
    
    def predict(self, test_data):
        """
        The test method, Runs the test data through each one of the trees and receives a prediction.
        For each tree, the test data is subsetted according to it`s corresponding indexes saved in
        <tree_columns> attribute. Finally, the final prediction for each sample is the mode value of the
        predictions it received from each one of the trees.
        Params:
            test_data (numpy array of m1*n): Contains all samples of the test data, including all features
        Returns:
            predictions (numpy array of m*1): the i`th value in this vector represents the prediction
                for the i`th sample in the test data.
        """
        predictions = np.zeros((len(test_data), len(self.bootstraps)))
        
        for i, tree in enumerate(self.trees):
            tree_data = test_data[:,self.tree_columns[i]]
            predictions[:,i] = tree.predict(tree_data).squeeze()
            
        final_predictions = (stats.mode(predictions, axis=1)[0]).reshape(len(test_data), 1)
        return final_predictions


def main():
    # Upload dataset, encode target var and split to train and test
    d = DataSet('../datasets/wdbc.data', label_col = 1, header=None)
    d.y[np.where(d.y == 'M')] = 1
    d.y[np.where(d.y == 'B')] = 0
    d.y = d.y.astype(np.int)
    d.trainTestSplit(80)
    
    # Use kfold cross validation to choose optimal number of bootstraps
    d.kFold(5)
    bootstrap_options = [5,10,15,20]
    accuracies = []
    
    for option in bootstrap_options:
        d.bootStrap(option)
        r = RandomForest()
        r.fit(d.bootstraps, 5, 3)
        r.predict(d.X_test)
        test_pred = r.predict(d.X_test)
        test_acc = np.mean(test_pred == d.y_test)
        accuracies.append(test_acc)
    
    optimal_bootstrap = bootstrap_options[np.array(accuracies).argmax()]
    print(f'The optimal number of bootstraps is: {optimal_bootstrap}')
    
    # Train final model
    d.bootStrap(optimal_bootstrap)
    r = RandomForest()
    r.fit(d.bootstraps, 5, 3)
    
    # Test
    r.predict(d.X_test)
    test_pred = r.predict(d.X_test)
    train_pred = r.predict(d.X_train)
    
    # Retreive and print measures
    test_acc = np.mean(test_pred == d.y_test)
    train_acc = np.mean(train_pred == d.y_train)
    print(f'Test Accuracy = {test_acc}')
    print(f'Train Accuracy = {train_acc}')


if __name__ == '__main__':
    main()
