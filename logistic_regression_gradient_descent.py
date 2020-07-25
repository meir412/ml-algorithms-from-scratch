import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

"""
Implementation of the logistic regression classification algorithm. Loss minimized
using gradient descent. 
The 'fit' function trains the model, and the 'predict' function is used to test the
model on new unlabeled data. The 'gradient' and 'loss' are called by 'fit' as part of the
training phase, and the 'sigmoid' function is used both in the train and in the test phases.
The 'trainTestSplit' function is used in the pre-proccessing stage in order to split the
data to train and test. The 'precision' function returns the precision metric on the prediction
vector against the label vector.
Simple use case of the algorithm is attached - classification of flowers from the iris dataset using
sepal length and width as features. The dataset is splitted and visualized before being used with
the 'fit' method to receive computed coefficients. The values of the loss during each iteration are
saved and then also visualized to validate conversion in the gradient descent optimization.
Pre received hyperparameters are used for the train phase. Predictions are then computed using the
learned coefficients, similarity measures are extracted and printed.
"""


def sigmoid(x):
    return 1/(1+ np.e**-x)


def loss(y, h):
    """
    Computes negative log likelihood which is the loss function for logistic 
    regression.
    Params:
        y - label vector of dimension (m,1)
        h - prediction vector of dimension (m,1)
    Returns:
        loss value (integer)
    """
    return -((y.T@np.log(h)+(1-y).T@np.log(1-h))/y.shape[0]).squeeze()


def gradient(y, h, X):
    """
    Computes gradient for negative log likelihood function (loss function) 
    according to the coefficients.
    Params:
       y - label vector of dimension (m,1)
       h - prediction vector of dimension (m,1)
       X - matrix containing n vectors of explanatory variables - dimensions(m,n)
    Returns:
        gradient - vector of (n,1)
    """
    return 1/(y.shape[0])*(X.T@(h-y))


def trainTestSplit(X, y, percentage):
    """
    Splits X and y into train and test according to given percentage.
    Params:
        X - matrix containing n vectors of explanatory variables - dimensions(m,n)
        y - label vector of dimension (m,1)
        percentage - percentage of data that should be allocated to train, integer
    Returns:
        Tuple of 4:
            X_train - subset of rows of X allocated for train 
            X_test - subset of rows of X allocated for test, compliment of X_train
            y_train - subset of y allocated for train, matching X_train
            y_test - subset of y allocated for train, matching X_test
    """
    
    m = X.shape[0]
    train_size = int(np.round((percentage/100)*m))
    train_indexes = np.random.choice(np.arange(m),train_size, replace=False)
    test_indexes = np.arange(m)[np.invert(np.isin(np.arange(m),train_indexes))]
    
    return (X[train_indexes], X[test_indexes], y[train_indexes], y[test_indexes])

    
def fit(X, y, theta_0, alpha, epochs):
    """
    Gradient descent algorithm for logistic regression, finds coefficients that
    will minimize the loss function and reach optimal prediction for binary values.
    Params:
        X - matrix containing n vectors of explanatory variables - dimensions(m,n)
        y - label vector of dimension (m,1)
        theta_0 - vector of (n,1) representing initial values of coefficients
        alpha - learning rate, determines step size in each iteration (double)
        epochs - number of iterations (integer)
    Returns:
        theta - final coefficient vector of size (n,1)
        loss_list - a list of size <epoch> containing the loss for each iteration
    """
    theta = theta_0
    loss_list = []
    
    for i in range(epochs):
        h = sigmoid(X@theta)
        l = loss(y, h)
        loss_list.append(l)
        grad = gradient(y, h, X)
        theta -= alpha*grad
    
    return (theta, loss_list)


def predict(test_samples, coefficients):
    """
    Multiply the coefficients received from the train phase with the samples and transform
    with the sigmoid function to receive the probabilities of a "1" label.
    Then convert the vector of probabilities to a vector of predicted classes, for binary
    classification with a hard coded threshold of 0.5.
    Params:
        test_samples (np array): matrix of m*n consisting of n feature values for m unlabeled samples
        coefficients (np array): vector of n*1 coefficients, one for each feature
    Returns:
        predictions (np array): vector of m*1 predicted labels - {0,1}
    """

    probabilities = sigmoid(test_samples@coefficients)
    predictions = probabilities.copy()
    predictions[probabilities >= 0.5] = 1
    predictions[probabilities < 0.5] = 0
    
    return predictions

    
def precision(predictions, labels):
    " Computes the precision metric between two binary vectors of the same size"
    true_positive = (labels + predictions == 2).sum()
    all_positives = labels.sum()
    precision = true_positive / all_positives
    return precision


iris = datasets.load_iris()

# Use the first two columns as X
X = pd.DataFrame(iris.data, columns=iris.feature_names)
X = X.iloc[:,:2]
m = X.shape[0]

# Labels will be decided this way:
# a. If the last column value is ‘Iris-setosa” - 1
# b. Else - 0
y = pd.Series(iris.target)
y = y.apply(lambda x: 1 if x==0 else 0)

# Create visualization of the data points
plt.scatter(X['sepal length (cm)'], X['sepal width (cm)'], c=y)
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.show()

# Add ones column for intercept in X
X = np.hstack((np.ones((m,1)),X))
n = X.shape[1]

# Initialize hyperparameters
alpha = 0.5
epochs = 1000
theta_0 = np.zeros((n,1)) + 0.1

# split to train & test
(X_train, X_test, y_train, y_test) = trainTestSplit(X,y.values,70)
y_train = y_train.reshape(y_train.size,1)
y_test = y_test.reshape(y_test.size,1)

# Train
(final_theta, loss_list) = fit(X_train, y_train, theta_0, alpha, epochs)

# Plot train loss as function of iterations
plt.plot(loss_list)
plt.title('Loss during training')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.show()

# Test
train_pred = predict(X_train, final_theta)
test_pred = predict(X_test, final_theta)

# Collect measures
train_accuracy = np.mean(y_train == train_pred)
test_accuracy = np.mean(y_test == test_pred)
train_precision = precision(train_pred, y_train)
test_precision = precision(test_pred, y_test)
test_loss = loss(y_test, sigmoid(X_test@final_theta))

# Print measures
print(f'train loss is {loss_list[-1]}')
print(f'test loss is {test_loss}')
print(f'train accuracy is {train_accuracy}')
print(f'test accuracy is {test_accuracy}')
print(f'train precision is {train_precision}')
print(f'test precision is {test_precision}')
print(f'final coefficients are: {final_theta}')










 