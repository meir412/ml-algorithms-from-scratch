# Support vector machine classifier - object oriented implementation

import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

from dataset import DataSet
from visualisation import plotLoss

class Svm():
    """
    Class for support vector machine classifier. Assumes a binary target variable of {-1,1}
    and numerical features. Uses Soft margin hinge loss as loss function and optimizes using gradient descent.
    The public methods are fit (train) and predict (test). The fit method uses the private methods
    __hingeLoss (to compute the loss) and __gradient (to compute the gradient).
    Attributes:
        alpha (float): A hyperparameter that determines the step size at each iteration (learning rate)
        C (float): A hyperparameter that determines how many data samples are allowed to be misclassified (regularization)
        w (numpy array of n*1): The learned parameters, represent the coefficients of the seperating hyperplane
        loss_list (list): the i`th item contains the loss value of the i`th iteration
    """
    def __init__(self):
        self.alpha = None
        self.C = None
        self.w = None
        self.loss_list = None
    
    
    def __hingeLoss(self, h, y):
        """
        Compute the soft margin hinge loss
        Params:
            h (numpy array of m*1): raw prediction vector, product of X and w
            y (numpy array of m*1): label vector
        Returns:
            loss (float): value of the soft margin hinge loss
        """        
        return 0.5*np.linalg.norm(self.w)**2 + self.C*np.sum(np.maximum(0, 1-y*h))
    
    
    def __gradient(self, h, y, X):
        """
        Compute the gradient of the soft margin hinge loss
        Params:
            h (numpy array of m*1): raw prediction vector, product of X and w
            y (numpy array of m*1): label vector
            X (numpy array of m*n): matrix containing all feature values for all samples
        Returns:
            gradient (numpy array of n*1): Each value represent the direction of steepest descent for a specific w
        """
        n = X.shape[1]
        gradient = np.zeros((n,1))
        
        for i,_ in enumerate(X):
            if h[i]*y[i] < 1:
                gradient += self.w + self.C*(-X[i,]*y[i]).reshape((n,1))
            else:
                gradient += self.w
        
        return gradient
            
        
    def fit(self, y, X, epochs, initial_w, alpha, C):
        """
        The training method of the model. Receives data, labels, and required hyperparameters.
        All hyperparamters are saved as attributes, including w which will be the learned paramters.
        The training is done using the gradient descent optimization technique. During each iteration,
        The loss value and gradient are computed, and the learned parameters are updated according to the
        gradient. Once the training is over, the w attribute contains the predicted parameters of the model.
        Params:
            y (numpy array of m*1): label vector
            X (numpy array of m*n): matrix containing all feature values for all samples
            epochs (int): number of iterations for gradient descent to run
            w (numpy array of n*1): Initial values for the parameters, represent the coefficients of the seperating hyperplane
            alpha (float): A hyperparameter that determines the step size at each iteration (learning rate)
            C (float): A hyperparameter that determines how many data samples are allowed to be misclassified (regularization)
        """
        self.loss_list = []
        self.w = initial_w
        self.alpha = alpha
        self.C = C
        
        for i in range(epochs):
            h = X@self.w
            loss = self.__hingeLoss(h, y)
            self.loss_list.append(loss)
            grad = self.__gradient(h, y, X)
            self.w -= self.alpha*grad
        


    def predict(self, X, y):
        """
        The test method of the model. multiplies the input test data with the learned parameters of
        the hyperplane (w) to acheive a positive or negative value for each instance. The positive
        values are then mapped to 1 and the negative values to -1. These are the final predictions.
        Params:
            X (numpy array of m*n): matrix containing all feature values for all samples of test data
            y (numpy array of m*1): label vector for test data
        Returns:
            predictions (numpy array of m*1): vector containing predicted class for each sample
        """
        raw_predictions = X@self.w
        predictions = np.copysign(np.ones_like(raw_predictions), raw_predictions)
        return predictions


def main():
    # Example use-case of the svm algorithm    
    d = DataSet('../datasets/diabetes.csv')
    d.scaleFeatures()  # standardize all columns
    d.addOnes()   # add ones col for intercept
    d.y[d.y == 0] = -1      # prep labels as {1,-1} for svm
    d.trainTestSplit(80)
    initial_w = np.random.random((d.X.shape[1],1)) - 0.5
    epochs = 20
    alpha = 0.001
    
    # Use kfold cross validation to choose optimal value for the soft margin hyperparameter
    d.kFold(5)
    C_options = [10**i for i in range(-5,2)]
    accuracies = []
    
    for option in C_options:
        
        fold_accuracies = []
        
        for fold in d.folds:
            s = Svm()
            s.fit(fold['train'][:,-1], fold['train'][:,:-1], epochs, initial_w, alpha, option)
            predictions = s.predict(fold['validation'][:,:-1], fold['y_val'])
            fold_accuracy = np.mean(predictions == fold['y_val'])
            fold_accuracies.append(fold_accuracy)
            
        accuracies.append(sum(fold_accuracies) / len(fold_accuracies))
    
    optimal_C = C_options[np.array(accuracies).argmax()]
    print(f'The optimal value for the soft margin hyperparameter: {optimal_C}')
    
    # Train and test final model
    svm = Svm()
    svm.fit(d.y_train, d.X_train, epochs, initial_w, alpha, optimal_C)
    plotLoss(svm.loss_list)
    train_pred = svm.predict(d.X_train, d.y_train)
    train_acc = (train_pred == d.y_train).mean()
    print(f'Train Accuracy = {train_acc}')
    test_pred = svm.predict(d.X_test, d.y_test)
    test_acc = (test_pred == d.y_test).mean()
    print(f'Test Accuracy = {test_acc}')


if __name__ == '__main__':
    main()