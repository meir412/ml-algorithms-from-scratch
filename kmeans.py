import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


class Kmeans():
    """
    Class representing the K-means clustering algorithm, including one of it`s variations - K-medians.
    The fit method is the only public method, the other 3 methods are private and called by the fit
    method. When calling the fit method, the user supplies the data and all the neccesary hyperparameters.
    The algorithm groups the data into a pre-determined number <k> of clusters, by finding the <k> means
    or medians with the optimal location according to the data.
    Attributes:
        k (int): hyperparameter, number of clusters data will be split into
        epsilon (float): hyperparameter, the algorithm will stop when the smallest movement of the means is
            less than epsilon
        measure (string): can be 'mean' (default) or 'median', determines which variation of the algorithm to run
        clusters (list): list of numpy arrays, the i`th array contains the data allocated to the i`th cluster, a subset
            of the rows of the original data
        means (np array of k,m): the i'th row represents the mean of the i`th cluster
        iterations (int): The number of iterations the algorithm has run at any stage
    """
    def __init__(self):
        self.k, self.epsilon, self.measure, self.clusters, self.means, self.iterations = None, None, None, None, None, None
   
    
    def fit(self, data, epsilon, k, measure='mean'):
        """
        The main clustering method. At first, picks 3 random points as the means (or medians), then:
        1. Assign each data point to a cluster according to its closest mean
        2. Recalculate the mean of each cluster
        3. If the movement of the means is larger than epsilon, return to 1
        Params:
            data (numpy array of m,n): the data intended for clustering
            epsilon (float): hyperparameter, the algorithm will stop when the smallest movement of the means is
                less than epsilon
            k (int): hyperparameter, number of clusters data will be split into
            measure (string): can be 'mean' (default) or 'median', determines which variation of the algorithm to run
        Returns:
            clusters (list): list of numpy arrays, the i`th array contains the data allocated to the i`th cluster, a subset
                of the rows of the original data
        """
        assert measure in ['mean', 'median'], 'Measure must be mean or median'
        
        self.epsilon = epsilon
        self.k = k
        self.measure = measure
        self.iterations = 0
        m,n = data.shape
        self.means = data[np.random.choice(range(m), k, replace = False),:]
        converged = False
        
        while converged == False:          
            self.__closest_means(data)
            new_means = self.__recalc_means()
            converged = self.__convergance(new_means)
            self.means = new_means
            self.iterations += 1
            
        return self.clusters

    
    def __closest_means(self, data):
        """
        For each point in the dataset, calculates the  closest mean (or median) according to the means
        saved in the <means> attribute. for means use l2 distance, for medians l1 distance. Update the <cluster>
        attribute accordingly (allocate each data point to the cluster formed by it's closest mean).
        Params:
            data (numpy array of m,n): the data intended for clustering
        """
        expanded_data = np.expand_dims(data,1)
        diff = expanded_data - self.means
        
        if self.measure == 'mean':
            distances = np.sqrt(np.sum(diff**2, axis=2))
        
        elif self.measure == 'median':
            distances = np.sum(np.abs(diff), axis=2)
            
        closest_mean = np.argmin(distances,axis=1)
        self.clusters = [data[np.where(closest_mean == i)[0],:]
                    for i in range(self.k)]
                
    
    def __recalc_means(self):
        " Recalculate the mean (or median) for each cluster according to the current allocated data points"
        
        if self.measure == 'mean':
            new_means = np.array([cluster.mean(axis=0) for cluster in self.clusters])
        elif self.measure == 'median':
            new_means = np.array([np.median(cluster, axis=0) for cluster in self.clusters])
        return new_means
    
    
    def __convergance(self, new_means):
        """
        Checks if the algorithm has reached convergance. Compares the means of the current iteration
        to the means of the previous iteration. The means of the previous iteration are stored in the
        <mean> attribute and the means of the current iteration are passed as a the <new_means> parameter.
        If all distances between each mean and it`s predeccesor are less than epsilon, the algorithm has
        converged and the method will return True, otherwise False. For means, the l2 distance measure is 
        used, for medians, the l1 distance is used.
        Params:
            new_means (np array of k,m): the i'th row represents the mean of the i`th cluster for the current iteration
        Returns:
            converged (boolean): True if algorithm has converged, otherwise False
        """
        converged = False
        diff = self.means - new_means
        if self.measure == 'mean':
            distances = np.sqrt(np.sum((diff)**2, axis=1))
        elif self.measure == 'median':
            distances = np.sum(np.abs(diff), axis=1)
        
        if (distances < self.epsilon).sum() == self.k:
            converged = True
        
        return converged
    
           
def main():
    """
    Example use case of the k-means algorithm. The iris dataset loaded from sklearn is used.
    A Kmeans object is created and it's fit method is called with hyperparameters of k = 3 and
    epsilon = 0.01. The computed clusters are stored in the <cluster> variable.
    """
    data = load_iris().data
    k = 3
    epsilon = 0.01
    kmeans = Kmeans()
    clusters = kmeans.fit(data, epsilon, k, 'mean')
  

if __name__ == '__main__':
    main()

