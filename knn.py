import numpy as np
from sklearn import datasets
import scipy

from dataset import DataSet


class KNN():
    """
    Class for the K Nearest Neighbors supervised learning algorithm. The class is initiated with a
    DataSet object which has already been split into train and test. Then in order to classify the test
    set, the user calls the predict method, which is the only public method for the class, the rest of the
    methods are called by it. The algorithm uses the euclidian distance (l2) as it's similarity measure,
    and the <k> hyperparameter, to determine the class of a test sample.
    Attributes:
        data_set (DataSet): the DataSet object must already contain data and must already be splitted
            into train and test (it's X_test, Xtrain, y_test, y_train attributes must be non-empty)
        distances (np array): contains the distance between each test sample (rows) to each train sample (columns)
        k_nearest (np array): each row represents a test sample and contains the indexes of the <k> nearest train
            samples (ordered)
        prediction (np array): contains the predicted class for each test sample
        accuracy (float): contains the accuracy value between the <prediction> and the dataset's <y_test>
    """
    def __init__(self, data_set):
        self.data_set = data_set
        self.distances, self.k_nearest, self.prediction, self.accuracy = None, None, None, None
    
    def __distance(self):
        """
        For each point in dataset1 calculate the euclidian distance to each point in dataset2.
        The 2 datasets can have a different number of rows, but must have the same
        number of columns.
        This is done by expanding the dimension of dataset1 to turn each of its rows
        into a seperate "matrix". Then numpy broadcasting is used to compute the difference
        between each of these matrixes and dataset2, then sum over axis2 to get distances.
        Datasets must contain numeric data.
        Params:
            dataset1: dataset of size m1*n
            dataset2: dataset of size m2*n
        Returns:
            distances: dataset of size m2*m1
        """
        dataset1 = self.data_set.X_test
        dataset2 = self.data_set.X_train
        dataset1 = np.expand_dims(dataset1,1)
        square_diff = np.square(dataset1 - dataset2)
        self.distances = np.sqrt(np.sum(square_diff, axis=2))
        
    
    def __nearestNeighbors(self, k):
        """
        For each point in the test set, finds the indexes of the k nearest neighbors
        from the train set, given a distance matrix of m*n.
        k must be <= n
        Params:
            distances: an array of m*n where each value represents the distance between
                a point in the train set (row) and a point in the test set (col)
            k: integer representing the number of nearest neighbors to return for 
                each row
        Returns:
            k_nearest: an array of m*k containing the indexes of the k nearest neighbors
                for each row
        """
        all_nearest = np.argsort(self.distances, axis=1)
        self.k_nearest = all_nearest[:,:k]
        
    
    def predict(self, k):
        """
        Predict the target class (categorical) for each row in the test set.
        Finds the target values of k nearest neighbors and chooses the mode value for
        each observation
        Params:
            k_nearest: an array of m*n containing the indexes of the k nearest neighbors
                for each row
            train_labels: an array of n*1 containing values of the target variable for
                each observation
        Returns:
            mode_label: an array of n*1 containing predictions of the target variable
        """
        assert k>=1, 'number of neighbors must be a positive integer'
        
        self.__distance()
        self.__nearestNeighbors(k)
        
        neighbor_labels = self.data_set.y_train[self.k_nearest].squeeze()
        
        if k==1:
            self.prediction = scipy.stats.mode(neighbor_labels)[0]
        else:
            self.prediction = scipy.stats.mode(neighbor_labels, axis=1)[0]
        self.__acc()
        
    
    def __acc(self):
        """
        Compute accuracy of predictions according to labels
        """
        self.accuracy = np.mean(self.prediction == self.data_set.y_test)


def main():
    """
    Example use-case of the algorithm on the iris dataset, loaded from sklearn. The predict method is called
    3 times with different values for <k>, and the accuracy for each call is printed.
    """
    iris = datasets.load_iris()
    d = DataSet()
    d.X = iris.data
    m = d.X.shape[0]
    d.y = iris.target.reshape(m,1)
    d.trainTestSplit(percentage=80)
    k = KNN(data_set = d)
    k.predict(k=1)
    print(f'Accuracy with k=1: {k.accuracy}')
    k.predict(k=2)
    print(f'Accuracy with k=2: {k.accuracy}')
    k.predict(k=10)
    print(f'Accuracy with k=10: {k.accuracy}')
    
    
if __name__ == '__main__':
    main()
    
    
