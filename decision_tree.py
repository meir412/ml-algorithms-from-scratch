# Decision tree classifier - object oriented implementation

import numpy as np
import pandas as pd
from scipy import stats

from dataset import DataSet


class DecisionTree():
    """
    Class for decision tree classifier. Assumes a binary target variable of {0,1},
    and numerical explanatory variables (not categorical). Public methods are 'fit' for the training stage
    and 'predict' for the test stage. The 'fit' method makes use of all the private methods:
    '__split', '__splitNode', '__getSplit', '__getSplitValue' and '__gini'.
    Attributes:
        root (Node): Will be assigned when fit is called. Represents the root of the
        decision tree, contains the labeled training data (numpy array), the split feature
        and value and pointers to 2 children nodes. The tree is built from the root node
        in the fit stage and is used to classify new samples in the predict stage.
        prune_value (int): The maximum depth the tree is allowed to grow to 
        random_subset (int): For random trees, the size of random subset of columns to choose
            for each split
    """
    def __init__(self):
        self.root = None
        self.prune_value = None
        self.random_subset = None
    
    
    def fit(self, dataset, prune_value, random_subset=None):
        """
        The public training method, receives the training data, creates the root node
        out of it, and then calls the recursive __spitNode method on the root, which in turn
        builds the whole tree.
        Params:
            dataset (numpy array): the 2d labeled training data, labels are assumed to be in the last column
            prune_value (int): The maximum depth the tree is allowed to grow to 
            random_subset (int): For random trees, the size of random subset of columns to choose
                for each split
        """
        self.random_subset = random_subset if random_subset else dataset.shape[1]-1
        self.prune_value = prune_value
        self.root = Node(None, None, None, None, 0, dataset)
        self.__splitNode(self.root)
        
    
    def predict(self, test_data):
        """
        The public test method, receives the test data. Every sample in the data is passed
        through the tree built in the training stage. Each node contains a feature and value, the sample`s
        value in this feature is compared to the node`s value, if the sample contains the larger (or equal) value
        it is sent to the right side of the tree, otherwise to the left. This continues until the sample
        reaches a leaf node which contains a prediction - this prediction is assigned to the feature.
        Params:
            test_data (numpy array of m*n): the 2d unlabeled test data
        Returns:
            predictions (numpy array of m*1): the i`th value in this vector represents the prediction
            for the i`th sample in the test data.
        """
        predictions = np.zeros((len(test_data), 1))
        
        for i,sample in enumerate(test_data):
            node = self.root
            
            while node.prediction is None:
                if sample[node.feature] >= node.split_val:
                    node = node.right_node
                else:
                    node = node.left_node
            
            predictions[i] = node.prediction
            
        return predictions
    
    
    def __splitNode(self, node):
        """
        Recursive method which is the heart of building the tree, called by the fit method
        in the training stage. For a given node, the method checks if the node`s data is "pure", i.e if
        all of the labels are 1 or all of the labels are 0, if so, the label is assigned to the node`s prediction
        attribute accordingly and the method returns. Another test is run to see if the tree has reached
        it`s maximum allowed depth, if so, the node`s prediction attribute is assigned with the label that is more
        common in the node`s dataset, then the method returns.
        If none of the above tests are true, the optimal feature and value are found and assigned to the node`s attributes,
        The dataset is then split according to these, and 2 new nodes are created with each subset of the data.
        These nodes are then assigned as children nodes of the current node, via it`s attributes.
        The method is then called (recursion) on each of these children nodes.
        Params:
            node (Node): when called contains only a dataset and a depth value
        """
        
        if node.dataset[:,-1].sum() == node.dataset[:,-1].size:
            node.prediction = 1
            return
        
        if node.dataset[:,-1].sum() == 0:
            node.prediction = 0
            return
        
        if node.depth > self.prune_value:
            node.prediction =  stats.mode(node.dataset[:,-1])[0][0]
            return
            
        else: 
            node.feature, node.split_val = self.__getSplit(node.dataset)
            right, left = self.__split(node.dataset, node.feature, node.split_val)
            node.right_node = Node(None,None,None,None,node.depth+1,right)
            node.left_node = Node(None,None,None,None,node.depth+1,left)
            self.__splitNode(node.right_node)
            self.__splitNode(node.left_node)
        
        
    def __getSplit(self, dataset):
        """
        Find the feature and value on which to split the dataset in order to acheive
        a minimum gini value. The method is called by the __splitNode as part of the training
        proccess. It iterates over all the features, gets the optimal split value for a feature
        and the corresponding gini value for the split using the __getSplitValue method. The method
        finds the minimum gini value and returns the corresponding feature and split value.
        If the tree was defined as a random tree, it will choose a random subset of columns to find the gini
        value for. If not, it will search through all columns.
        Params:
            dataset (numpy array of m*n): the 2d labeled training data
        Returns:
            feature (int): index of optimal feature to split on
            split_val (float): optimal value to split on feature
        """
        
        n = dataset.shape[1]-1
        columns = np.random.choice(np.arange(n), self.random_subset, replace=False)
        
        gini_values = np.ones((self.random_subset,2))
        
        for i,col in enumerate(columns):
            gini_values[i,:] = self.__getSplitValue(dataset, col)
                
        min_index = gini_values[:,1].argmin()
        split_val = gini_values[min_index,0]
        feature = columns[min_index]
        
        return (feature, split_val)
    
    
    def __split(self, dataset, feature, split_value):
        """
        Given a dataset, feature, and value for this feature, split the dataset into 2 new datasets,
        Where the first contains only samples where the feature`s value is greater or equal to the input value,
        And the second contains those that are lesser than the input value.
        Params:
            dataset (numpy array of m*n): the 2d labeled training data
            feature (int): index of feature to split on
            split_value (float): value to split on feature
        Returns:
            sub1 (numpy array of m1*n): 2d labeled training data (subset of input data)
            sub2 (numpy array of m2*n): 2d labeled training data (subset of input data)
        """ 
        sub1 = dataset[dataset[:,feature] >= split_value]
        sub2 = dataset[dataset[:,feature] < split_value]
        return (sub1, sub2)
    
    
    def __getSplitValue(self, dataset, feature):
        """
        For a specific feature (numerical) find splitting point in the feature that will split the
        data optimally, i.e achieve a minimum gini value. Sort the values of the feature and find the
        'mid_values' - the average between each 2 consecutive sorted values. Get the gini index for the
        split using the __gini method and return the minimum gini value and it`s corresponding value
        of the feature. The method is called by __getSplit as part of the traininf proccess.
        Params:
            dataset (numpy array of m*n): the 2d labeled training data
            feature (int): index of optimal feature to split on
        Returns:
            split_value (float): optimal value to split on feature
            gini_value (float): corresponding gini value of the split
        """
        sorted_indexes = dataset[:,feature].argsort()
        sorted_v = dataset[:,feature][sorted_indexes]
        mid_values = np.array([sorted_v[i] + (sorted_v[i+1]-sorted_v[i])/2 
                               for i in range(len(sorted_v)-1)])
        
        gini_values = np.ones_like(mid_values)
        
        for i,value in enumerate(mid_values):
            set1, set2 = self.__split(dataset, feature, value) 
            gini_v = self.__gini(set1, set2)
            gini_values[i] = gini_v
        
        return (mid_values[gini_values.argmin()], gini_values.min())
        
    
    def __gini(self, dataset1, dataset2):
        """
        Return gini value for a split (2 datasets) with a binary target variable. The gini value for
        each dataset is {1 - the sum of square probabilities to randomly get each class}. The gini
        value for both datasets is the weighted average of the 2 gini values, according to the size of
        each dataset.
        Receives:
           dataset1 (numpy array of m1*n): 2d labeled training data (binary target)
           dataset2 (numpy array of m2*n): 2d labeled training data (binary target)
        Returns:
            final_gini (float): gini value for the 2 input datasets
        """
        size1 = len(dataset1)
        size2 = len(dataset2)
        total_size = size1 + size2
        target1 = len(dataset1[dataset1[:,-1] == 1])
        target2 = len(dataset2[dataset2[:,-1] == 1])
        gini1 = (1 - ((target1/size1)**2 + (1-(target1/size1))**2)) if size1 > 0 else 0
        gini2 = 1 - ((target2/size2)**2 + (1-(target2/size2))**2) if size2 > 0 else 0
        final_gini = (size1/total_size)*gini1 + (size2/total_size)*gini2

        return final_gini
                
                
class Node():
    """
    Class representing a node, which in this context is used as part of a decision tree.
    Each node represents either a decision node, deciding how to split the data, or a leaf/terminal node
    which contains a prediction value.
    Attributes:
        right_node (Node): if node isn't a leaf, it will have a right child, which is also a node
            containing the subset of the data received after the split (greater than the split).
        left_node (Node): if node isn't a leaf, it will have a right child, which is also a node
            containing the subset of the data received after the split (lesser than the split).
        feature (int): index of feature to split on
        split_val (float): value to split on feature
        depth (int): The length between the root of the tree to the node
        dataset (numpy array): the subset of labeled training data that is received from the parent node`s split
        prediction (int): Only for leaves, predicted class (target) for this subset of the data    
    """
    def __init__(self, right_node, left_node, feature, split_val, depth, dataset, prediction=None):
        self.right_node = right_node
        self.left_node = left_node
        self.feature = feature
        self.split_val = split_val
        self.depth = depth
        self.dataset = dataset
        self.prediction = prediction


def main():
    # Example usage of the DecisionTree class
    # Prepare dataset, encode labels and split to train and test
    d = DataSet('../datasets/wdbc.data', label_col = 1, header=None)
    d.y[np.where(d.y == 'M')] = 1
    d.y[np.where(d.y == 'B')] = 0
    d.y = d.y.astype(np.int)
    d.trainTestSplit(80)
    
    # # find optimal prune value with cross validation
    # d.kFold(5)
    # prune_options = [2,4,6,8]
    # accuracies = []
    
    # for option in prune_options:
        
    #     fold_accuracies = []
        
    #     for fold in d.folds:
    #         t = DecisionTree()
    #         t.fit(fold['train'], option)
    #         predictions = t.predict(fold['validation'])
    #         fold_accuracy = np.mean(predictions == fold['y_val'])
    #         fold_accuracies.append(fold_accuracy)
            
    #     accuracies.append(sum(fold_accuracies) / len(fold_accuracies))
    
    # optimal_prune = prune_options[np.array(accuracies).argmax()]
    optimal_prune = 6
    random_subset = 6
    
    # Train final model, test and print accuracy
    t = DecisionTree()
    # t.fit(d.train, optimal_prune, random_subset)
    t.fit(d.train, optimal_prune, random_subset)
    test_pred = t.predict(d.X_test)
    train_pred = t.predict(d.X_train)
    
    test_acc = np.mean(test_pred == d.y_test)
    train_acc = np.mean(train_pred == d.y_train)
    print(f'The tree was pruned at depth: {optimal_prune}')
    print(f'Test Accuracy = {test_acc}')
    print(f'Train Accuracy = {train_acc}')


if __name__ == '__main__':
    main()
