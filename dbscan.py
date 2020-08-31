import numpy as np
from sklearn.datasets import load_iris


class Dbscan():
    """
    Class representing the dbscan (density based spatial clustering of applications) unsupervised
    learning algorithm. The fit method is the only public method, the other 4 are private methods
    called by it. When calling the fit method, the user supplies the data to be clustered and the 
    algorithm's hyperparameters. The method returns a list of numpy arrays, each containing a different
    cluster, except the first array, which contains all the 'outlier' points which don't belong to any
    of the clusters. The number of returned clusters varied according to data and hyperparameters.
    Attributes:
        epsilon (float): hyperparamter, the radius around a point to search for neighbors
        min_points (int): hyperparameter, the minimum number of neighbors to declare the point as core point
        cluster_vec (numpy array of m,1): the i`th value represents the cluster number that the i`th point
            has ben allocated to. A value of -1 means the point is an outlier
        current_cluster (int): represents the cluster that is currently being expanded in the algorithm
    """  
    def __init__(self):
        self.epsilon, self.min_points, self.distance_mat = None, None, None
    
    
    def fit(self, data, epsilon, min_points):
        """
        The main method that is called by the user and returns the clustered data.
        Assigns all points to the "-1" cluster (outliers) and calls the __distance_matrix to compute all
        distances between points. Iterates over each point in the dataset, if it is an outlier, the 
        __expand_cluster method is called on it, to see if it can be expanded into a cluster, if the method
        returns true, the <current_cluster> variable is incremented, and a new cluster search begins.
        Params:
            data (numpy array of m,n): the data intended for clustering
            epsilon (float): hyperparamter, the radius around a point to search for neighbors
            min_points (int): hyperparameter, the minimum number of neighbors to declare the point as core point
        Returns:
            cluster_list (list): each element is a numpy array representing a cluster, the array in the first element
                represents the outliers, the points that are not part of any cluster
        """
        self.epsilon = epsilon
        self.min_points = min_points
        m,n = data.shape
        self.cluster_vec = np.ones(m) - 2  # class vector initiated with -1 sentinel value (all outliers)
        self.current_cluster = 0
        self.__distance_matrix(data)

        for i in range(m):
            if self.cluster_vec[i] == -1:
                if self.__expand_cluster(i):
                    self.current_cluster +=1
        
        n_clusters = len(np.unique(self.cluster_vec))  # final number of clusters including the outlier cluster
        clusters_list = [data[np.where(self.cluster_vec == i)[0],:]
                for i in range(-1, n_clusters-1)]

        return clusters_list
        
    
    def __expand_cluster(self, point_id):
        """
        The method receives a specific point and checks if it can be expanded into a cluster, called by
        the fit method. First, checks if the input point is a core points (has no less than <min_points> in its
        <epsilon> radius), if it isn't a core, return false, if it is, start expanding the cluster.
        Once a point is determined a core, the <seeds> variable, is used as a queue to add new points to the cluster.
        Initially, <seeds> contains the neighbors of the input point. Then, while <seeds> isn't empty, the first point is assigned
        the value of the current cluster and its neighbors of are searched for, if there are enough neighbors, the unclassified ones
        are added to the queue. After expanding the cluster the method returns true.
        Params:
            point_id (int): the index of the point that starts expanding the cluster
        Returns:
            expanded (boolean): true if the initial point can be expanded, false otherwise
        """
        seeds = self.__region_query(point_id)
        
        if len(seeds) < self.min_points:
            return False
        
        else:          
            while len(seeds) > 0:
                current_point = seeds[0]
                self.cluster_vec[current_point] = self.current_cluster
                neighbors = self.__region_query(current_point)
                
                if len(neighbors) >= self.min_points:
                    for n in neighbors:
                        if self.cluster_vec[n] == -1:
                            seeds.append(n)
                
                seeds = seeds[1:]   # pop first value out of the queue
                
            return True
        
            
    def __region_query(self, point_index):
        """
        For a specific point, returns it's 'neighbors', i.e the number of points that are under <min_distance>
        away from the input point.
        Params:
            point_id (int): the index of the point that starts expanding the cluster
        Returns:
            neighbors (list): list of ids of the input point's neighbors
        """
        point_distances = self.distance_mat[point_index]
        neighbors = np.where(point_distances < self.epsilon)[0]
        neighbors = neighbors.tolist()
        return neighbors


    def __distance_matrix(self, data):
        """
        Compute distance matrix (distance between each point to each point).
        Params:
            data (numpy array of m,n): the data intended for clustering
        """
        expanded_data = np.expand_dims(data,1)
        diff = expanded_data - data
        self.distance_mat = np.sqrt(np.sum(diff**2, axis=2))
    

def main():
    """
    Example use case of the algorithm using the iris dataset loaded from sklearn and pre-defined
    hyperparameters.
    """
    data = load_iris().data
    epsilon = 0.5
    min_points = 6
    dbscan = Dbscan()
    clusters = dbscan.fit(data, epsilon, min_points)


if __name__ == '__main__':
    main()
