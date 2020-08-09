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
    tree for each dataset. The classifier also uses random trees, i.e trees that for each split, select a random
    subset of features to choose the best split for. In the prediction stage, each test sample receives a prediction from
    each tree, and the mode value is selected as the final prediction. Attributes are assigned in the fit method.
    Attributes:
        bootstraps (list): Each value is a numpy array of m0*n, a bootstrap sample of the original train data
        random_subset (int): The number of features to be sampled that the tree will randomly choose from
            for each split.
        trees (list): Each value is a DecisionTree that is part of the forest
        prune_value (int): the maximum depth that the trees are allowed to grow to
    """
    
    def __init__(self):
        self.bootstraps = None
        self.trees = None
        self.prune_value = None
        self.random_subset = None
    
    
    def fit(self, bootstraps, random_subset, prune_value):
        """
        For each bootstrap, the method creates a tree and trains it on this bootstrap.
        The trees are saved in the <trees> attribute.
        Params:
            bootstraps (list): Each value is a numpy array of m*n, a bootstrap sample of the original train data
            random_subset (int): The number of features to be sampled that the tree will randomly choose from
                for each split.
            prune_value (int): the maximum depth that the trees are allowed to grow to
        """
        self.bootstraps = bootstraps
        self.prune_value = prune_value
        self.random_subset = random_subset
        self.trees = []
        n = bootstraps[0].shape[1] - 1
        
        for i,strap in enumerate(bootstraps):
            t = DecisionTree()
            t.fit(strap, self.prune_value, self.random_subset)
            self.trees.append(t)
            
        
        
    
    def predict(self, test_data):
        """
        The test method, Runs the test data through each one of the trees and receives a prediction.
        For each tree. The final prediction for each sample is the mode value of the predictions it received from each one of the trees.
        Params:
            test_data (numpy array of m1*n): Contains all samples of the test data, including all features
        Returns:
            predictions (numpy array of m*1): the i`th value in this vector represents the prediction
                for the i`th sample in the test data.
        """
        predictions = np.zeros((len(test_data), len(self.bootstraps)))
        
        for i, tree in enumerate(self.trees):
            predictions[:,i] = tree.predict(test_data).squeeze()
            
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
    random_subset = 5
    
    # Train final model
    d.bootStrap(optimal_bootstrap)
    r = RandomForest()
    r.fit(d.bootstraps, random_subset, 3)
    
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
