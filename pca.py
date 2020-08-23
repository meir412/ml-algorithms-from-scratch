# Principle component analysis - object oriented implementation

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2 as cv
from sklearn.preprocessing import StandardScaler

from dataset import DataSet
from visualisation import show_images

class Pca():
    """
    Class for principal component analysis algorithm. Assumes a data matrix  of m,n where each row
    is a sample and each column is a feature. The algorithm allows the user to perform dimensionality
    reduction on the data, i.e represent the data with a lesser number of features. During the fit method,
    the principle components of the data are found and arranged by importance. Once the model is trained,
    data can be projected onto the principal components (usually reducing it's dimension) using the transform method.
    Transformed (reduced) data can be projected back onto the original axes of the data using the inverse_transform method.
    Attributes:
        components (numpy array of n,k): The truncated components after calculation, each component as a column
        n_components (int): The number of components to be left after reduction
        explained_variance (1d numpy array of m): The variance explained by each component
        explained_variance_ratio (1d numpy array of m): The ratio of variance explained by each component
        means (numpy array of 1,m): The mean value for each original feature
    """
    def __init__(self):
        self.components = None
        self.n_components = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        self.means = None
    
    
    def fit(self, X, n_components = None, explained_variance = None):
        """
        The train method, extract the principal components using the eigenvectors and eigenvalues of the data's covariance 
        matrix. The eigenvectors (evecs) are computed either by diagonalising the covariance matrix itself (if m>n)
        or by using the covariance of the transpose and multiplication by the data (in order to decrease complexity).
        The eigenvectors are sorted according to the eigenvalues. The user can determine the magnitude of the reduction
        by using the <n_components> or <explained_variance> variables to leave only a subset of the eigenvectors, only
        one of these parameters should be specified, if none are specified, the full set of eigenvectors is returned.
        The eigenvectors that are left are stored as an attribute for later use by the transform and inverse_transform methods.
        Params:
            X (numpy array of m,n): The training data, with rows representing samples
            n_components (int): The number of principle components to leave after truncation
            explained_variance (float): The height of the explained_variance_ratio to be left after truncation,
                determines the number of components which will be left
        """
        assert n_components == None or explained_variance == None, "Specify number of components OR explained variance"   
        m = X.shape[0]
        n = X.shape[1]
        self.means = X.mean(axis=0).reshape(1,n)
        centered_X = X - self.means
        
        if n>m:
            # Use transpose trick
            trans_cov = centered_X @ centered_X.T
            evals, evecs = np.linalg.eig(trans_cov)
            evecs = evecs[:,np.flip(evals.argsort())]
            evecs = centered_X.T @ evecs
            
        else:
            # Use regular covariance matrix
            cov = centered_X.T @ centered_X
            evals, evecs = np.linalg.eig(trans_cov)
            evecs = evecs[:,np.flip(evals.argsort())]            
        
        evals = evals.astype(np.float)
        self.explained_variance = np.flip(np.sort(evals))
        self.explained_variance_ratio = evals / np.sum(evals)
        evecs = evecs.astype(np.float)
        evecs = (evecs / np.linalg.norm(evecs, axis=0))  # Divide each eigenvector by its norm
        
        if explained_variance != None:
            cummulative_variance = np.cumsum(self.explained_variance_ratio)
            length = len(cummulative_variance[cummulative_variance < explained_variance])
            self.n_components = length+1
            self.components = evecs[:,:length+1]
            
        elif n_components != None:
            self.n_components = n_components
            self.components = evecs[:,:self.n_components]
        
        else:
            self.n_components = n
    
    
    def transform(self, X):
        """
        Using the transform method, the dimension of features can be reduced for any number
        of samples using the means and the eigenvectors calculated during the train stage.
        Params:
            X (numpy array of m1,n): Data to be transformed, any number of samples, same number of
                features as the training data
        Returns:
            transformed_data (numpy array of m1,k): Transformed data with less features
        """
        return (X - self.means) @ self.components
    
    
    def inverse_transform(self, X):
        """
        Using the inverse_transform method, data that has already been transformed to the lesser dimension
        can be transformed back to the original dimension of the data.
        Params:
            X (numpy array of m1,k): Data to be transformed back, any number of samples, number of features
                must be according to output of transform method
        Returns:
            reconstructed_data (numpy array of m1, n): The reconstructed data, with the original number of features
        """
        return X @ self.components.T + self.means
    

def main():
    """
    Use case of pca for dimensionality reduction on face images (the eigenfaces technique).
    Use the face database found here: http://cswww.essex.ac.uk/mv/allfaces/faces94.html
    Use first 1500 images for training and then use the trained model to reduce and reconstruct
    a new image that wasn't part of the test. Plot the eigenfaces (the eigenvectors) received from training.
    Plot the original image and the reconstructed one side by side.
    """
    # Load and flatten images
    images = []
    
    for root, dirs, files in os.walk("../datasets/faces94", topdown=False):
        for name in files:
            if os.path.splitext(name)[1] != '.jpg':
                continue
            image = cv.imread(os.path.join(root, name))
            image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
            image = image.reshape(image.size)
            images.append(image)
    
    # Images as rows, subset first 1500 for training and different 1 for test    
    test_image = np.array(images)[1500,:].reshape(1,36000)
    images = np.array(images)[:1600,:]
    
    # Train, reduce and reconstruct new image
    pca = Pca()
    pca.fit(images, explained_variance = 0.95)
    test_reduced = pca.transform(test_image)
    test_reconstructed = pca.inverse_transform(test_reduced)
    
    # Plot eigenfaces
    eigenfaces = [pca.components[:,i].reshape(200,180) for i in range(12)]
    show_images(eigenfaces, cols=3)
    
    # Plot
    show_images([test_image.reshape(200,180), test_reconstructed.reshape(200,180)],
                titles = ['Original', 'Reconstructed'])


if __name__ == '__main__':
    main()