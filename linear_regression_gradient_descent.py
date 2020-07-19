"""
Linear regression problems solved with different versions of the gradient
descent algorithm.

A simple dataset of 3 observations, 1 explanatory variable and 1 target variable
is manually created. To the explanatory variable, we add an all-ones vector for
the bias and an additional variable which is the squares of the first variable.
Hyperparameters for initial coefficients and number of iterations are declared.

The `gradientDescent` function implements the gradient descent algorithm, it makes
use of the 2 functions `computeLoss` and `gradient`.
The function is called with different learning rates, and the loss value is 
plotted against the number of iterations to check that the loss is indeed
converging. An experiment is run to find the optimal learning rate (alpha).

The `momentumGradientDescent` and `nesterovGradientDescent` functions are implementations
of gradient descent with different optimization techniques for updating the coefficients.
The functions are then called to check how the convergance of the loss is affected.
"""

import numpy as np
import matplotlib.pyplot as plt


# Creation and preperation of data
data = np.array([[0,1,2],[1,3,7]]).T
n = data.shape[0]
X = np.hstack((np.ones((n,1)), data[:,0].reshape(n,1), ((data[:,0])**2).reshape(n,1)))
y = data[:,-1].reshape(n,1)

# Initialize hyperparameters
theta_0 = np.array([[2,2,0]]).T
iterations = 100


def computeLoss(h, y):
    return np.mean((h-y)**2)

def gradient(h, X, y, theta):
    return X@(h-y)
    
def gradientDescent(X, y, alpha, iterations, theta_0):
    """
    Function that runs the gradient descent algorithm for linear regression.
    Params:
        X - 2d array representing explanatory data, each column represents
            a different feature, size is m*n
        y - 2d array of size m*1 representing labels corresponding to X
            alpha - float representing the step size multiplier for each iteration
        iterations - integer representing number of iterations to run
        theta_0 - initial values for the coefficients predicted by the linear
                  regression, for first iteration, 2d array of n*1
    Return:
        tuple containing theta and loss list
        theta - final values for predicted coefficients, 2d array of n*1
        loss_list - list of loss values at each step of the iteration, list
                    of size iterations.
    """
    theta = theta_0
    loss_list = []
    
    for i in range(iterations):        
        h = X @ theta
        loss = computeLoss(h, y)
        loss_list.append(loss)
        grad = gradient(h, X, y, theta)
        theta = theta - alpha*grad
    
    return (theta, loss_list)

# Run gradient descent for three specified alphas, print final losses and
# graphs showing loss value against number of iterations for each alpha.
alphas = [0.01, 0.1, 1.0]

for i,alpha in enumerate(alphas):
    (theta, loss_list) = gradientDescent(X, y, alphas[i], iterations, theta_0)
    print(f'for alpha={alpha}, final loss is: {loss_list[-1]}, final thetas are: {theta}')
    plt.plot(range(100), loss_list)
    plt.title(f'loss for alpha={alpha}')
    plt.show()

# Find optimal alpha, should be between 0.1 and 0.12
alphas = np.arange(0.1,0.12,0.002)
losses = np.zeros((alphas.size, 2))

for i,alpha in enumerate(alphas):
    (theta, loss_list) = gradientDescent(X, y, alphas[i], iterations, theta_0)
    losses[i] = [np.round(alpha,3), loss_list[-1]]

min_loss = np.min(losses[:,1])
min_alpha = losses[np.argmin(losses[:,1]), 0]
print(f'Minimum loss of {min_loss} found for alpha = {min_alpha}')

'''
With a=0.01, the convergence is too slow and not sufficient after 100 iters.
With a=0.1, the convergence is good and reaches a value of around 10^-9.
With a=1, the convergence is perfect after 1 iter, but this is a very rare case
because of our initial values and thetas. 
Optimal alpha is ~ 0.11
'''

# Momentum gradient descent

def momentumGradientDescent(X, y, alpha, iterations, theta_0, lamda):
    
    theta = theta_0
    previous_velocity = 0
    loss_list = []
    
    for i in range(iterations):        
        h = X @ theta
        loss = computeLoss(h, y)
        loss_list.append(loss)
        grad = gradient(h, X, y, theta)
        velocity = lamda*previous_velocity + alpha*grad
        theta = theta - velocity
        previous_velocity = velocity
    
    return (theta, loss_list)

lamda = 0.9
alpha = 0.01

(theta, loss_list) = momentumGradientDescent(X, y, alpha, iterations, theta_0, lamda)
print(f'with alpha 0.01 and momentum lambda 0.9, final loss is: {loss_list[-1]}, final thetas are: {theta}')
plt.plot(range(100), loss_list)
plt.title(f'loss for alpha={alpha} momentum sgd')
plt.show()

# Nesterov accelerated gradient descent

def nesterovGradientDescent(X, y, alpha, iterations, theta_0, lamda):
    
    theta = theta_0
    velocity = 0
    loss_list = []
    
    for i in range(iterations):        
        h = X @ theta + lamda*velocity
        loss = computeLoss(h, y)
        loss_list.append(loss)
        grad = gradient(h, X, y, theta)
        velocity = lamda*velocity + alpha*grad
        theta = theta - velocity
    
    return (theta, loss_list)

lamda = 0.9
alpha = 0.01

(theta, loss_list) = nesterovGradientDescent(X, y, alpha, iterations, theta_0, lamda)
print(f'with alpha 0.01 and nesterov lambda 0.9, final loss is: {loss_list[-1]}, final thetas are: {theta}')
plt.plot(range(100), loss_list)
plt.title(f'loss for alpha={alpha} nesterov sgd')
plt.show()
    








