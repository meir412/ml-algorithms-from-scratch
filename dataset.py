import numpy as np
import pandas as pd


class DataSet():
    """
    Class for dataset object. Initiate a new DataSet object with a path or url for csv data.
    The dataset will automatically be splitted to X (explanatory variables) and y (target variable),
    assuming that the target variable is found in the last column. After initiating, it is possible
    to split the dataset into train and test sets using the class `trainTestSplit` method.
    """   

    def __init__(self, path, label_col = -1, header='infer'):
        """
        Initiate new DataSet instance, load dataset as a pandas dataframe, split into X
        (features), and y (target).
        **note: all _train and _test subsets are initiated as None and must be extracted using
        the trainTestSplit method.
        params:
            path (string): local path or url of dataset
            label_col (int): column index of the labels of the dataset
            header (string): Assign None in order to load data that doesn't contain column titles
        attributes:
            X (numpy array): contains all of the dataset excluding the target variable,
                shape: m*n
            y (numpy array): contains only the target variable, shape m*1
            column_names (pandas index): contains all column names
            X_train - subset of rows of X allocated for train 
            X_test - subset of rows of X allocated for test, compliment of X_train
            y_train - subset of y allocated for train, matching X_train
            y_test - subset of y allocated for train, matching X_test
            train - X_train and y_train in one dataset, the column for y is the last column
            folds (list): list of [validation, train] subsets of the train data
        """
        
        data = pd.read_csv(path, header=header)
            
            
        data = pd.read_csv(path)
        self.X = data.drop(data.columns[label_col], axis=1).values
        m = self.X.shape[0]
        self.y = data.iloc[:,label_col].values.reshape((m,1))
        self.column_names = data.columns
        self.X_train, self.X_test, self.y_train, self.y_test, self.train, self.folds = None,None,None,None,None,None
    
    
    
    def trainTestSplit(self, percentage):
        """
        Splits X and y into train and test according to given percentage.
        Assigns the test and train subsets into the _train and _test attributes of the instance.
        Params:
            percentage - percentage of data that should be allocated to train, integer
        """
        
        m = self.X.shape[0]
        train_size = int(np.round((percentage/100)*m))
        train_indexes = np.random.choice(np.arange(m),train_size, replace=False)
        test_indexes = np.arange(m)[np.invert(np.isin(np.arange(m),train_indexes))]
        self.X_train = self.X[train_indexes]
        self.X_test = self.X[test_indexes]
        self.y_train = self.y[train_indexes]
        self.y_test = self.y[test_indexes]
        self.train = np.column_stack((self.X_train, self.y_train))
    
    
    def kFold(self, k):
        """
        Split train dataset to multiple validation and train sets for cross validation.
        Use the kfold methodology, split the dataset into k even validation sets, for each validation
        set, take the rest of the data as it`s training set. Prepare a list of these validation and train
        subsets and assign to the folds attribute.
        THIS METHOD CAN ONLY BE CALLED AFTER TRAINTESTSPLIT METHOD IS CALLED
        params:
            k (int): the desired number of folds
        """
        
        m = self.train.shape[0]
        fold_size = int(m/k)
        self.folds = []
        
        for i in range(k):
            validation_indexes = np.random.choice(np.arange(m),fold_size, replace=False)
            train_indexes = np.arange(m)[np.invert(np.isin(np.arange(m), validation_indexes))]
            train = self.train[train_indexes]
            validation = self.train[validation_indexes]
            y_val = self.y_train[validation_indexes]
            self.folds.append({'train':train, 'validation':validation, 'y_val': y_val})


def main():

# Usage example of the dataset object
    d = DataSet('../../datasets/wdbc.data', label_col = 1, header=None)
    d.trainTestSplit(80)
    d.kFold(5)


if __name__ == '__main__':
    main()