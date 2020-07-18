"""
Linear regression problems solved using the normal equations
Sample 2d data, one explanatory variable and one target variable, initially the target variable should be completely
dependant on the explanatory variable, with a known equation, linear or polynomial.
After sampling add gaussian noise to the target variable.
Conduct 3 experiments:
1. Sample 10 data points which have a linear correlation (no bias)
2. Sample 10 data points which have a linear correlation (with bias)
3. Sample 10 data points which have a polynomial correlation (with bias)
For each experiment use the normal equations to predict the coefficients describing the correlation.
For each experiment plot the line formed by the predicted coefficients against the actual data points in order
to visualize the experiment.
"""

import numpy as np
import matplotlib.pyplot as plt


# Sample points for the explanatory variable from the uniform[-20,20] distribution
v2 = np.random.uniform(-20, 20, 10)

# Create target variable according to linear equation y=6x
y_vector = v2*6

# add noise to target variable from standard gaussian distribution
y_vector += np.random.normal(0, 1, 10)

# Bind both vectors in one array
first_array = np.concatenate((v2.reshape((10,1)), y_vector.reshape(10,1)), 1)


# Repeat data creation process to simulate linear correlation plus bias
v5 = np.random.uniform(-20, 20, 10)
second_y = v5*3 + 4  # according to linear equation y=3x+4
second_array = np.concatenate((v5.reshape(10,1), second_y.reshape(10,1)), 1)
second_array[:,1] += np.random.normal(0, 1, 10)

# Repeat data creation process to simulate polynomial correlation
v6 = np.random.uniform(-20, 20, 10)
third_y = v6**2 + 4*v6 + 3 # according to equation x^2 + 4x + 3
third_array = np.concatenate((v6.reshape(10,1), third_y.reshape(10,1)), 1)
third_array[:,1] += np.random.normal(0, 1, 10)


# Find best fit for linear coefficients according to normal equation (first experiment)
X = first_array[:,0].reshape(10,1)
y = first_array[:,1].reshape(10,1)
h = (np.linalg.inv(X.T@X))@(X.T@y)

# Plot predicted line against actual labels (first experiment)
line_x = np.linspace(-20,20,3).reshape(3,1)
line_y = line_x * h
plt.plot(first_array[:,0], first_array[:,1], 'ro' )
plt.plot(line_x, line_y)
plt.show()


# Find best fit for linear coefficients according to normal equation (second experiment)
# Add an [1,...,1] vector to the explanatory vectors in order to find the bias (the free coefficient)
X = np.concatenate((np.ones(10).reshape(10,1), second_array[:,0].reshape(10,1)),1)
y = second_array[:,1].reshape(10,1)
h2 = (np.linalg.inv(X.T@X))@(X.T@y)


# Plot predicted line against actual labels (second experiment)
line_x = np.linspace(-20,20,3).reshape(3,1)
line_y = h2[0] + line_x * h2[1]
plt.plot(second_array[:,0], second_array[:,1], 'ro' )
plt.plot(line_x, line_y)
plt.show()

# Find best fit for polynomial coefficients according to normal equation (third experiment)
# Add an [1,...,1] vector to the explanatory vectors in order to find the bias (the free coefficient)
x_0 = np.ones(10).reshape(10,1)
x_1 = third_array[:,0].reshape(10,1)

# Assuming a linear correlation between the square of the explanatory variable and the target
# we will add the square as an additional explanatory variable and use the normal equations to find the coefficient
x_2 = (third_array[:,0]**2).reshape(10,1)
X = np.hstack((x_0,x_1,x_2))
y = third_array[:,1].reshape(10,1)
h3 = (np.linalg.inv(X.T@X))@(X.T@y)

# Plot predicted polynomial line against actual labels (third experiment)
line_x = np.linspace(-20,20,20).reshape(20,1)
line_y = h3[0] + h3[1] * line_x + h3[2] * ((line_x)**2)
plt.plot(line_x, line_y)
plt.plot(third_array[:,0], third_array[:,1], 'ro' )
plt.show()
