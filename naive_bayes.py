# Naive bayes classifier - object oriented implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from dataset import DataSet


class GaussianNaiveBayes():
    """
    Class for the Gaussian Naive Bayes classifier. Assumes normal
    distribution for all feature vectors, also assumes independence between features.
    No parameters are needed for initiating a new classifier. After initiation use the
    `fit` method with train data and the GaussianNaiveBayes instance will update its attributes
    according to the training. The `predict` method can then be called with unlabeled test data
    to return a vector of predictions.
    Attributes:
        total_summary (pd dataframe): table of mean and std for each feature in the train data
        class_summaries (list): a list of size k containing pandas dataframes, each representing a
            class, with means and stds for each feature
        class_priors (list): a list of size k containing the prior probability of each class
    """
    
    def __init__(self):     
        self.total_summary, self.class_summaries, self.class_priors = None, None, None
    
    
    def fit(self, X_train, y_train):
        """
        Fit (train) function for the Gaussian Naive Bayes classifier. Assumes normal
        distribution for all feature vectors. In order to receive predictions, the classifier
        uses the likelihoods of features, given a certain class from the target. To compute these
        likelihoods under the normal distribution assumption, the mean and std for each feature
        given a certain class are computed in the training phase. The bayesian classifier also
        uses the prior probabilites of each of the classes to predict, these are also computed
        in the training phase. The function iterates over each class and extracts these values.
        Params:
            train_data (pandas dataframe) - all of the train data including the target with k classes.
        Returns:
        """
        train_df = pd.DataFrame(np.hstack((X_train, y_train)))
        self.total_summary = train_df.iloc[:,:-1].describe().loc[['mean','std'],:]
        self.class_summaries = []
        self.class_priors = []
        
        for i in np.unique(train_df.iloc[:,-1]):
            class_data = train_df[train_df.iloc[:,-1] == i].drop(train_df.columns[-1], axis=1)
            class_prior = class_data.shape[0] / train_df.shape[0]
            self.class_priors.append(class_prior)
            class_summary = class_data.describe().loc[['mean','std'],:]
            self.class_summaries.append(class_summary)
            
    
    def predict(self, test_data):
        """
        Predict (test) function for the Gaussian Naive Bayes classifier. Assumes normal distribution
        for all feature vectors, also assumes that the features are independent. For each test sample,
        compute the posterior probability of each class and assign the sample to the class with the maximum
        probability. The posterior probabilities are computed using Bayes' theorem: (likelihood*prior)/evidence.
        The priors of each class were computed in the train stage and are input to the test. The evidence is
        the product of evidences for each feature, and is computed using the normal density function and
        the overall mean and std for each feature, given as input. The likelihood is computed in the same way,
        but uses the mean and std of the feature, given a certain class. Iterate over the classes to receive
        posterior probability for each class, for each sample.
        Params:
            test_data (numpy array): vector of m test samples with n features, size must be m*n, if it
                is one sample, should be a 2d row vector of m*1
            summaries (list): a list of size k containing pandas dataframes, each representing a
                class, with means and stds for each feature
            priors (list): a list of size k containing the prior probability of each class
            total_summary (pandas df) - dataframe of 2*n containing mean and std of each feature, given 
                the whole dataset
        Returns:
            predictions (numpy array): 2d vector of size m*1 with the predicted class for each test sample
        """
        posteriors = []
        total_means = self.total_summary.iloc[0,:]
        total_stds = self.total_summary.iloc[1,:]
        feature_evidences = stats.norm.pdf(test_data,total_means, total_stds)
        evidence = np.prod(feature_evidences, axis=1)
    
        for index,summary in enumerate(self.class_summaries):
            class_means = summary.iloc[0,:]
            class_stds = summary.iloc[1,:]
            feature_likelihoods = stats.norm.pdf(test_data, class_means, class_stds)
            likelihood = np.product(feature_likelihoods, axis=1)
            posterior = self.class_priors[index] * (likelihood / evidence)
            posteriors.append(posterior)
        
        posteriors = np.array(posteriors).T
        predictions = np.argmax(posteriors, axis=1).reshape((test_data.shape[0],1))
        return predictions


def main():
    d = DataSet('../datasets/diabetes.csv')
    d.trainTestSplit(80)
    g = GaussianNaiveBayes()
    g.fit(d.X_train, d.y_train)
    predictions = g.predict(d.X_test)
    accuracy = np.mean(predictions == d.y_test)
    print(f'Accuracy: {accuracy}')

if __name__ == '__main__':
    main()
